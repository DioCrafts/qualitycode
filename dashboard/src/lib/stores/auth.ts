import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import type { User, Organization, DashboardPermissions } from '$lib/types';

interface AuthState {
  user: User | null;
  organization: Organization | null;
  permissions: DashboardPermissions | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: Error | null;
}

interface LoginCredentials {
  email: string;
  password: string;
}

interface TokenResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

function createAuthStore() {
  const { subscribe, set, update } = writable<AuthState>({
    user: null,
    organization: null,
    permissions: null,
    isAuthenticated: false,
    loading: false,
    error: null
  });

  // Token management
  let accessToken: string | null = null;
  let refreshToken: string | null = null;
  let tokenExpiryTimeout: ReturnType<typeof setTimeout> | null = null;

  function setTokens(tokens: TokenResponse) {
    accessToken = tokens.accessToken;
    refreshToken = tokens.refreshToken;

    if (browser) {
      // Store refresh token securely (httpOnly cookie would be better)
      sessionStorage.setItem('refreshToken', refreshToken);
    }

    // Set up token expiry handling
    if (tokenExpiryTimeout) {
      clearTimeout(tokenExpiryTimeout);
    }

    tokenExpiryTimeout = setTimeout(() => {
      refreshAccessToken();
    }, (tokens.expiresIn - 60) * 1000); // Refresh 1 minute before expiry
  }

  async function refreshAccessToken() {
    if (!refreshToken) {
      logout();
      return;
    }

    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken })
      });

      if (!response.ok) {
        throw new Error('Token refresh failed');
      }

      const tokens = await response.json();
      setTokens(tokens);
    } catch (error) {
      console.error('Token refresh failed:', error);
      logout();
    }
  }

  async function login(credentials: LoginCredentials) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials)
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      setTokens(data.tokens);

      update(state => ({
        ...state,
        user: data.user,
        organization: data.organization,
        permissions: data.permissions,
        isAuthenticated: true,
        loading: false
      }));

      return data;
    } catch (error) {
      update(state => ({
        ...state,
        error: error as Error,
        loading: false
      }));
      throw error;
    }
  }

  async function logout() {
    try {
      if (accessToken) {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear tokens
      accessToken = null;
      refreshToken = null;
      if (browser) {
        sessionStorage.removeItem('refreshToken');
      }

      if (tokenExpiryTimeout) {
        clearTimeout(tokenExpiryTimeout);
      }

      // Reset state
      set({
        user: null,
        organization: null,
        permissions: null,
        isAuthenticated: false,
        loading: false,
        error: null
      });
    }
  }

  async function checkAuth() {
    if (!browser) return;

    const storedRefreshToken = sessionStorage.getItem('refreshToken');
    if (!storedRefreshToken) {
      return;
    }

    update(state => ({ ...state, loading: true }));

    try {
      const response = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${storedRefreshToken}`
        }
      });

      if (!response.ok) {
        throw new Error('Auth check failed');
      }

      const data = await response.json();
      
      if (data.tokens) {
        setTokens(data.tokens);
      }

      update(state => ({
        ...state,
        user: data.user,
        organization: data.organization,
        permissions: data.permissions,
        isAuthenticated: true,
        loading: false
      }));
    } catch (error) {
      update(state => ({
        ...state,
        error: error as Error,
        loading: false,
        isAuthenticated: false
      }));
    }
  }

  async function updateProfile(updates: Partial<User>) {
    if (!accessToken) throw new Error('Not authenticated');

    try {
      const response = await fetch('/api/auth/profile', {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`
        },
        body: JSON.stringify(updates)
      });

      if (!response.ok) {
        throw new Error('Profile update failed');
      }

      const updatedUser = await response.json();
      
      update(state => ({
        ...state,
        user: updatedUser
      }));

      return updatedUser;
    } catch (error) {
      console.error('Profile update failed:', error);
      throw error;
    }
  }

  async function switchOrganization(organizationId: string) {
    if (!accessToken) throw new Error('Not authenticated');

    try {
      const response = await fetch('/api/auth/switch-org', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`
        },
        body: JSON.stringify({ organizationId })
      });

      if (!response.ok) {
        throw new Error('Organization switch failed');
      }

      const data = await response.json();
      
      update(state => ({
        ...state,
        organization: data.organization,
        permissions: data.permissions
      }));

      return data;
    } catch (error) {
      console.error('Organization switch failed:', error);
      throw error;
    }
  }

  // Attach auth header to all API requests
  if (browser) {
    const originalFetch = window.fetch;
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      if (accessToken && typeof input === 'string' && input.startsWith('/api/')) {
        init = init || {};
        init.headers = {
          ...init.headers,
          'Authorization': `Bearer ${accessToken}`
        };
      }
      return originalFetch(input, init);
    };
  }

  return {
    subscribe,
    login,
    logout,
    checkAuth,
    updateProfile,
    switchOrganization,
    
    getAccessToken: () => accessToken,
    
    hasPermission: (permission: keyof DashboardPermissions) => {
      const state = get({ subscribe });
      return state.permissions?.[permission] || false;
    },
    
    reset() {
      logout();
    }
  };
}

export const authStore = createAuthStore();

// Derived stores
export const currentUser = derived(
  authStore,
  $auth => $auth.user
);

export const currentOrganization = derived(
  authStore,
  $auth => $auth.organization
);

export const userRole = derived(
  authStore,
  $auth => $auth.user?.role || null
);

export const canManageProjects = derived(
  authStore,
  $auth => $auth.permissions?.canManageProjects || false
);

export const canViewAnalytics = derived(
  authStore,
  $auth => $auth.permissions?.canViewAnalytics || false
);

export const canConfigureSecurity = derived(
  authStore,
  $auth => $auth.permissions?.canConfigureSecurity || false
);

// Initialize auth check on load
if (browser) {
  authStore.checkAuth();
}
