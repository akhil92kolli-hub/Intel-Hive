# IntelHive website

Responsive React landing page for IntelHive. It uses Vite to produce a static `dist/` build, so the same site can be deployed to Cloudflare Pages or Vercel without a server or framework adapter.

## Requirements

- Node.js 20.19+ (or 22.12+)
- npm

## Run locally

```sh
npm ci
npm run dev
```

Open the local URL printed by Vite. To verify the production build locally:

```sh
npm run build
npm run preview
```

## Deploy to Cloudflare Pages

Connect the repository to Cloudflare Pages and set:

- **Build command:** `npm run build`
- **Build output directory:** `dist`
- **Root directory:** `/`
- **Node.js version:** `22`

The repository includes `wrangler.jsonc` for the Pages output directory and `public/_headers` for production response headers. A build can also be uploaded with `npx wrangler pages deploy dist` after `npm run build`.

## Deploy to Vercel

Import the repository into Vercel. It detects Vite automatically; use:

- **Build command:** `npm run build`
- **Output directory:** `dist`
- **Install command:** `npm ci`
- **Node.js version:** `22.x`

`vercel.json` provides the site's security headers.

## Before launch

- The early-access call to action is intentionally disabled. It does not accept, transmit, or store email addresses; connect a waitlist provider before enabling signups.
- The device app is not available. The privacy section describes items to resolve before launch and does not claim unimplemented protections are in place.
- Earnings and device activity in the visuals are illustrative, not a promise of income or live network data. Validate economics and update the example before publishing.
- Review all copy, links, policies, and security headers for the production domain before launch.
