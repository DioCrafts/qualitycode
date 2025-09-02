import { browser } from '$app/environment';
import type { Notification, RealTimeMessage } from '$lib/types';
import { derived, get, writable } from 'svelte/store';

interface RealTimeState {
    connected: boolean;
    connectionType: 'sse' | 'websocket' | null;
    lastHeartbeat: number | null;
    messages: RealTimeMessage[];
    notifications: Notification[];
    error: Error | null;
}

interface RealTimeConfig {
    url: string;
    reconnectInterval: number;
    maxReconnectAttempts: number;
    heartbeatInterval: number;
}

const defaultConfig: RealTimeConfig = {
    url: '/api/realtime',
    reconnectInterval: 5000,
    maxReconnectAttempts: 5,
    heartbeatInterval: 30000
};

function createRealTimeStore() {
    const { subscribe, set, update } = writable<RealTimeState>({
        connected: false,
        connectionType: null,
        lastHeartbeat: null,
        messages: [],
        notifications: [],
        error: null
    });

    let eventSource: EventSource | null = null;
    let websocket: WebSocket | null = null;
    let reconnectAttempts = 0;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    let heartbeatInterval: ReturnType<typeof setInterval> | null = null;

    function cleanup() {
        if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
        }

        if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
        }

        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        if (websocket) {
            websocket.close();
            websocket = null;
        }
    }

    function handleMessage(data: any) {
        try {
            const message: RealTimeMessage = JSON.parse(data);

            update(state => ({
                ...state,
                messages: [...state.messages, message].slice(-100), // Keep last 100 messages
                lastHeartbeat: Date.now()
            }));

            // Handle different message types
            switch (message.type) {
                case 'notification':
                    update(state => ({
                        ...state,
                        notifications: [...state.notifications, message.payload as Notification]
                    }));
                    break;

                case 'analysis_update':
                    // Dispatch custom event for other stores to listen
                    if (browser) {
                        window.dispatchEvent(new CustomEvent('analysis:update', {
                            detail: message.payload
                        }));
                    }
                    break;

                case 'security_alert':
                    if (browser) {
                        window.dispatchEvent(new CustomEvent('security:alert', {
                            detail: message.payload
                        }));
                    }
                    break;
            }
        } catch (error) {
            console.error('Failed to parse realtime message:', error);
        }
    }

    function connectSSE(config: RealTimeConfig) {
        if (!browser) return;

        try {
            eventSource = new EventSource(config.url);

            eventSource.onopen = () => {
                update(state => ({
                    ...state,
                    connected: true,
                    connectionType: 'sse',
                    error: null
                }));
                reconnectAttempts = 0;

                // Start heartbeat monitoring
                heartbeatInterval = setInterval(() => {
                    const state = get({ subscribe });
                    const now = Date.now();
                    if (state.lastHeartbeat && now - state.lastHeartbeat > config.heartbeatInterval * 2) {
                        // Connection seems dead, reconnect
                        reconnect(config);
                    }
                }, config.heartbeatInterval);
            };

            eventSource.onmessage = (event) => {
                handleMessage(event.data);
            };

            eventSource.onerror = (error) => {
                console.error('SSE error:', error);
                update(state => ({
                    ...state,
                    connected: false,
                    error: new Error('SSE connection failed')
                }));
                cleanup();
                reconnect(config);
            };

        } catch (error) {
            console.error('Failed to create SSE connection:', error);
            update(state => ({
                ...state,
                error: error as Error
            }));
        }
    }

    function connectWebSocket(config: RealTimeConfig) {
        if (!browser) return;

        try {
            const wsUrl = config.url.replace(/^http/, 'ws');
            websocket = new WebSocket(wsUrl);

            websocket.onopen = () => {
                update(state => ({
                    ...state,
                    connected: true,
                    connectionType: 'websocket',
                    error: null
                }));
                reconnectAttempts = 0;

                // Send heartbeat ping
                heartbeatInterval = setInterval(() => {
                    if (websocket?.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({ type: 'ping' }));
                    }
                }, config.heartbeatInterval);
            };

            websocket.onmessage = (event) => {
                handleMessage(event.data);
            };

            websocket.onclose = () => {
                update(state => ({
                    ...state,
                    connected: false
                }));
                cleanup();
                reconnect(config);
            };

            websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                update(state => ({
                    ...state,
                    connected: false,
                    error: new Error('WebSocket connection failed')
                }));
            };

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            update(state => ({
                ...state,
                error: error as Error
            }));
        }
    }

    function reconnect(config: RealTimeConfig) {
        if (reconnectAttempts >= config.maxReconnectAttempts) {
            update(state => ({
                ...state,
                error: new Error('Max reconnection attempts reached')
            }));
            return;
        }

        reconnectAttempts++;
        reconnectTimeout = setTimeout(() => {
            connect(config);
        }, config.reconnectInterval * reconnectAttempts);
    }

    function connect(config: RealTimeConfig = defaultConfig) {
        cleanup();

        // Try WebSocket first, fallback to SSE
        if (browser && 'WebSocket' in window) {
            connectWebSocket(config);
        } else {
            connectSSE(config);
        }
    }

    return {
        subscribe,

        connect,

        disconnect() {
            cleanup();
            update(state => ({
                ...state,
                connected: false,
                connectionType: null
            }));
        },

        sendMessage(message: any) {
            if (websocket?.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify(message));
                return true;
            }
            return false;
        },

        clearNotifications() {
            update(state => ({
                ...state,
                notifications: []
            }));
        },

        markNotificationRead(id: string) {
            update(state => ({
                ...state,
                notifications: state.notifications.map(n =>
                    n.id === id ? { ...n, read: true } : n
                )
            }));
        },

        reset() {
            cleanup();
            set({
                connected: false,
                connectionType: null,
                lastHeartbeat: null,
                messages: [],
                notifications: [],
                error: null
            });
        }
    };
}

export const realTimeStore = createRealTimeStore();

// Derived stores
export const connectionStatus = derived(
    realTimeStore,
    $realtime => ({
        connected: $realtime.connected,
        type: $realtime.connectionType,
        lastHeartbeat: $realtime.lastHeartbeat
            ? new Date($realtime.lastHeartbeat).toLocaleTimeString()
            : 'Never'
    })
);

export const unreadNotifications = derived(
    realTimeStore,
    $realtime => $realtime.notifications.filter(n => !n.read)
);

export const notificationCount = derived(
    unreadNotifications,
    $unread => $unread.length
);

// Auto-connect when in browser
if (browser) {
    realTimeStore.connect();
}
