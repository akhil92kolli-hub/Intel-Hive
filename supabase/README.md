# IntelHive Supabase backend

Supabase provides two narrow services for the Android and iOS workers:

- append-only mobile benchmark ingestion through `public.benchmark_results`;
- public model manifests and, when the project permits the object size, GGUF
  artifacts through the `model-artifacts` Storage bucket.

It does not store scheduler state, assignments, activations, or KV caches.

## Database

Apply the migrations in `migrations/` with the Supabase CLI or the project
deployment workflow. The benchmark table uses Row Level Security. Anonymous and
authenticated app clients may insert rows for the pinned model/version/digest,
but cannot select, update, or delete them. Administrative reporting must use a
trusted server-side credential outside the mobile app.

The public publishable key is safe to include in an APK because authorization
is enforced by grants, constraints, and RLS. The same rule applies to the iOS
app bundle. Never commit a personal access token, secret key, or service-role
key.

## Model artifacts

The Android default manifest is available at:

```text
https://uozyxansakogtpqxcpdp.supabase.co/storage/v1/object/public/model-artifacts/manifests/qwen2.5-3b-instruct/1.0.0/manifest.json
```

The reserved GGUF object path is:

```text
model-artifacts/qwen2.5-3b-instruct/1.0.0/qwen2.5-3b-instruct-q4_k_m.gguf
```

The artifact is 2,104,932,768 bytes. The project's current 50 MB global Storage
limit must be raised before uploading it. Keep the manifest's upstream download
URL in place until the Supabase object has been uploaded and its full SHA-256
has been verified. For the large upload, use Supabase resumable TUS uploads and
the direct Storage hostname rather than a single standard upload request.

After uploading, update the manifest's `download_url`, upload the revised
manifest, and verify the public object byte size and digest before releasing an
APK that depends on it.
