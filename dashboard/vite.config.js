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
				target: 'http://backend:8000',
				changeOrigin: true
			},
			'/api/realtime': {
				target: 'ws://backend:8000',
				ws: true
			}
		}
	},
	define: {
		'process.env.PUBLIC_API_URL': JSON.stringify('http://backend:8000'),
		'process.env.PUBLIC_WS_URL': JSON.stringify('ws://backend:8000')
	}
});
