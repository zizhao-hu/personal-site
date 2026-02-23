import { createRequire } from 'module';
const require = createRequire(import.meta.url);

/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
        unoptimized: false,
    },
    transpilePackages: ['lucide-react'],
    webpack: (config, { isServer }) => {
        if (!isServer) {
            // pptxgenjs references node:fs, node:https etc.
            config.resolve.fallback = {
                ...config.resolve.fallback,
                fs: false,
                https: false,
                http: false,
                net: false,
                tls: false,
                child_process: false,
                stream: false,
                zlib: false,
                path: false,
                os: false,
                crypto: false,
            };

            // Handle node: protocol scheme by stripping the prefix
            // so resolve.fallback entries take effect
            const webpack = require('webpack');
            config.plugins.push(
                new webpack.NormalModuleReplacementPlugin(
                    /^node:/,
                    (resource) => {
                        resource.request = resource.request.replace(/^node:/, '');
                    }
                )
            );
        }
        return config;
    },
};

export default nextConfig;
