<script lang="ts">
  import { page } from '$app/stores';
  import { userRole } from '$lib/stores';
  import type { UserRole } from '$lib/types';
  
  export let sidebarOpen = true;
  export let mobile = false;
  
  // Navigation items based on user role
  const navigationItems = [
    {
      icon: '📊',
      label: 'Dashboard',
      href: '/',
      roles: ['developer', 'tech_lead', 'manager', 'security', 'qa'] as UserRole[]
    },
    {
      icon: '📁',
      label: 'Proyectos',
      href: '/projects',
      roles: ['developer', 'tech_lead', 'manager'] as UserRole[]
    },
    {
      icon: '🔍',
      label: 'Análisis',
      href: '/analysis',
      roles: ['developer', 'tech_lead', 'qa'] as UserRole[]
    },
    {
      icon: '🛡️',
      label: 'Seguridad',
      href: '/security',
      roles: ['security', 'tech_lead', 'manager'] as UserRole[]
    },
    {
      icon: '📈',
      label: 'Métricas',
      href: '/metrics',
      roles: ['tech_lead', 'manager', 'qa'] as UserRole[]
    },
    {
      icon: '⚙️',
      label: 'Configuración',
      href: '/settings',
      roles: ['developer', 'tech_lead', 'manager', 'security', 'qa'] as UserRole[]
    }
  ];
  
  // Filter navigation items based on user role
  $: filteredNavItems = navigationItems.filter(item => 
    !$userRole || item.roles.includes($userRole)
  );
  
  // Check if route is active
  function isActive(href: string): boolean {
    if (href === '/') {
      return $page.url.pathname === '/';
    }
    return $page.url.pathname.startsWith(href);
  }
</script>

<nav class="sidebar-nav" class:collapsed={!sidebarOpen && !mobile}>
  <div class="logo-container">
    <div class="logo">
      {#if sidebarOpen || mobile}
        <span class="logo-text">🐜 CodeAnt</span>
      {:else}
        <span class="logo-icon">🐜</span>
      {/if}
    </div>
  </div>
  
  <ul class="nav-list">
    {#each filteredNavItems as item}
      <li>
        <a 
          href={item.href}
          class="nav-item"
          class:active={isActive(item.href)}
          title={!sidebarOpen && !mobile ? item.label : ''}
        >
          <span class="nav-icon">{item.icon}</span>
          {#if sidebarOpen || mobile}
            <span class="nav-label">{item.label}</span>
          {/if}
        </a>
      </li>
    {/each}
  </ul>
  
  <div class="sidebar-footer">
    {#if sidebarOpen || mobile}
      <div class="version-info">
        <small>v1.0.0</small>
      </div>
    {/if}
  </div>
</nav>

<style>
  .sidebar-nav {
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: 1rem 0;
  }
  
  .logo-container {
    padding: 0 1rem 2rem;
    border-bottom: 1px solid var(--color-border, #e5e5e5);
    margin-bottom: 1rem;
  }
  
  .logo {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 48px;
    font-size: 1.5rem;
    font-weight: bold;
  }
  
  .logo-text {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .logo-icon {
    font-size: 2rem;
  }
  
  .nav-list {
    flex: 1;
    list-style: none;
    padding: 0;
    margin: 0;
  }
  
  .nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    color: var(--color-text-secondary, #666);
    text-decoration: none;
    transition: all 0.2s ease;
    position: relative;
  }
  
  .nav-item:hover {
    background-color: var(--color-bg-hover, #f0f0f0);
    color: var(--color-text-primary, #333);
  }
  
  .nav-item.active {
    color: var(--color-primary, #3b82f6);
    background-color: var(--color-primary-light, #eff6ff);
  }
  
  .nav-item.active::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background-color: var(--color-primary, #3b82f6);
  }
  
  .nav-icon {
    font-size: 1.25rem;
    width: 1.5rem;
    text-align: center;
    flex-shrink: 0;
  }
  
  .nav-label {
    font-size: 0.875rem;
    font-weight: 500;
  }
  
  .sidebar-footer {
    padding: 1rem;
    border-top: 1px solid var(--color-border, #e5e5e5);
    text-align: center;
    color: var(--color-text-muted, #999);
  }
  
  /* Collapsed state */
  .collapsed .logo-container {
    padding: 0 0.5rem 1rem;
  }
  
  .collapsed .nav-item {
    justify-content: center;
    padding: 0.75rem 0.5rem;
  }
  
  .collapsed .nav-icon {
    font-size: 1.5rem;
  }
  
  /* Dark mode */
  :global(.dark) .logo-container {
    border-bottom-color: var(--color-border-dark, #333);
  }
  
  :global(.dark) .nav-item {
    color: var(--color-text-secondary-dark, #aaa);
  }
  
  :global(.dark) .nav-item:hover {
    background-color: var(--color-bg-hover-dark, #2a2a2a);
    color: var(--color-text-primary-dark, #fff);
  }
  
  :global(.dark) .nav-item.active {
    background-color: var(--color-primary-dark, #1e40af);
  }
  
  :global(.dark) .sidebar-footer {
    border-top-color: var(--color-border-dark, #333);
  }
  
  /* High contrast */
  :global(.high-contrast) .nav-item {
    font-weight: 600;
  }
  
  :global(.high-contrast) .nav-item.active {
    background-color: #000;
    color: #fff;
  }
  
  /* Reduced motion */
  :global(.reduced-motion) .nav-item {
    transition: none !important;
  }
</style>
