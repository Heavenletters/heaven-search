# Integrating with KeystoneJS + Astro

This guide covers how to embed Heavenletters search into your upgraded website.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Astro (SSG/SSR)                                │
│  ┌─────────────────┐  ┌───────────────────────┐ │
│  │ /heavenletters   │  │ /api/heaven-search    │ │
│  │ Search UI page   │──│ (Astro API route)     │ │
│  └─────────────────┘  └──────────┬────────────┘ │
└──────────────────────────────────┼──────────────┘
                                   │ fetch
                          ┌────────▼────────────┐
                          │  Docker container    │
                          │  heaven-search:8000  │
                          │  (on same VPS)       │
                          └─────────────────────┘
```

## Option A: Astro API Route (recommended for SSR)

Create `src/pages/api/search.ts`:

```typescript
// src/pages/api/search.ts
import type { APIRoute } from "astro";

const SEARCH_API = "http://localhost:8000";
const API_KEY = import.meta.env.HEAVEN_API_KEY;

export const GET: APIRoute = async ({ request, url }) => {
  const q = url.searchParams.get("q");
  const mode = url.searchParams.get("mode") || "hybrid";
  const top = url.searchParams.get("top") || "10";

  const resp = await fetch(
    `${SEARCH_API}/search?q=${encodeURIComponent(q!)}&mode=${mode}&top=${top}`,
    {
      headers: { Authorization: `Bearer ${API_KEY}` },
    }
  );

  return new Response(await resp.text(), {
    status: resp.status,
    headers: { "Content-Type": "application/json" },
  });
};
```

Then on your search page:

```astro
---
// src/pages/heavenletters/search.astro
const query = Astro.url.searchParams.get("q") || "";
let results = [];
if (query) {
  const resp = await fetch(
    `${Astro.url.origin}/api/search?q=${encodeURIComponent(query)}`
  );
  results = (await resp.json()).results;
}
---
<form method="get">
  <input name="q" value={query} placeholder="Search Heavenletters..." />
  <button>Search</button>
</form>

{results.map((r) => (
  <article>
    <h3>{r.title || `HL #${r.external_id}`}</h3>
    <small>Score: {r.score} ({r.source})</small>
    <p>{r.content.slice(0, 300)}...</p>
  </article>
))}
```

## Option B: Build-time indexing (SSG)

For a fully static site, pre-index the Heavenletters:

```typescript
// During build: fetch all results for common queries and embed them
// In astro.config.mjs or a build script
```

But SSG limits you to pre-defined queries. The API route approach (Option A) is better
for free-form research queries.

## Option C: Client-side (SPA mode)

```astro
---
// Astro island — runs in browser
---
<script>
  async function search(q) {
    const resp = await fetch(
      `/api/search?q=${encodeURIComponent(q)}`
    );
    return (await resp.json()).results;
  }
</script>
```

## KeystoneJS Integration

If you want search results surfaced inside Keystone's admin or as a GraphQL field:

```typescript
// keystone.ts — custom query resolver
import { graphql } from "@keystone-6/core";

export const extendGraphqlSchema = graphql.extend((base) => {
  return {
    query: {
      heavenletterSearch: graphql.field({
        type: graphql.list(graphql.JSON),
        args: {
          query: graphql.arg({ type: graphql.nonNull(graphql.String) }),
          top: graphql.arg({ type: graphql.Int, defaultValue: 10 }),
        },
        async resolve(_source, { query, top }, _context) {
          const resp = await fetch(
            `http://localhost:8000/search?q=${encodeURIComponent(query)}&top=${top}`,
            {
              headers: { Authorization: `Bearer ${process.env.HEAVEN_API_KEY}` },
            }
          );
          const data = await resp.json();
          return data.results;
        },
      }),
    },
  };
});
```

## The `/analyze` endpoint

For your "analyze this article through the Capacities lens" use case:

```typescript
// POST /api/analyze
const resp = await fetch("http://localhost:8000/analyze", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    query: "creating community through shared purpose",
    instructions:
      "Reference these Heavenletters and identify themes of unity, " +
      "collective consciousness, and divine communion. " +
      "Note any cognitive or perceptual shifts being described.",
    top_k: 5,
    model: "deepseek-chat",
  }),
});
```

## CORS & Security

In production, set `CORS_ORIGINS` to your website domain:

```env
CORS_ORIGINS=https://your-heavenletters-site.com
```

Put the search container on the same Docker network as your Keystone/Astro app
and don't expose port 8000 publicly — only the Astro API route faces the internet.
