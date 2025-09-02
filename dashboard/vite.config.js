import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	preview: {
		allowedHosts: ['qualitycode.duckdns.org', 'localhost']
	},
	server: {
		proxy: {
			'/api': {
				target: 'http://localhost:8001',
				changeOrigin: true
			},
			'/api/realtime': {
				target: 'ws://localhost:8001',
				ws: true
			}
		}
	},
	define: {
		'process.env.PUBLIC_API_URL': JSON.stringify('http://localhost:8001'),
		'process.env.PUBLIC_WS_URL': JSON.stringify('ws://localhost:8001')
	}
});
