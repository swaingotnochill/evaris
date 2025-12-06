-- Evaris Database Initialization
-- This script runs on PostgreSQL container startup
--
-- Sets up:
-- - Required extensions
-- - Row Level Security (RLS) for multi-tenancy
-- - App configuration variables for RLS context
--
-- Note: Prisma handles schema creation via migrations.
-- This file only sets up extensions and RLS policies.

-- ============================================================================
-- Extensions
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Configuration Variables for RLS
-- ============================================================================

-- Create custom parameter for current organization (used by RLS policies)
-- This allows us to set/reset org context per-connection

-- Set default value to empty string
ALTER DATABASE evaris_test SET app.current_organization_id = '';
ALTER DATABASE evaris_test SET app.is_admin = 'false';

-- ============================================================================
-- RLS Policies (applied after Prisma migrations)
-- ============================================================================

-- Note: These policies are created after Prisma runs migrations.
-- Run the following SQL after `prisma migrate deploy`:
--
-- Enable RLS on tables:
--   ALTER TABLE "organization" ENABLE ROW LEVEL SECURITY;
--   ALTER TABLE "project" ENABLE ROW LEVEL SECURITY;
--   ALTER TABLE "Eval" ENABLE ROW LEVEL SECURITY;
--   ALTER TABLE "test_result" ENABLE ROW LEVEL SECURITY;
--   ALTER TABLE "Trace" ENABLE ROW LEVEL SECURITY;
--   ALTER TABLE "Span" ENABLE ROW LEVEL SECURITY;
--   ALTER TABLE "Log" ENABLE ROW LEVEL SECURITY;
--
-- Create policies:
--   CREATE POLICY org_isolation ON "Eval"
--     USING (
--       current_setting('app.is_admin', true) = 'true'
--       OR "organizationId" = current_setting('app.current_organization_id', true)
--     );
--
-- Similar policies for other tables...

-- ============================================================================
-- Test Data (Optional - only for development/testing)
-- ============================================================================

-- Create a test organization and project if they don't exist
-- This is useful for running E2E tests without prior setup

-- Insert test organization (only if not exists)
INSERT INTO "organization" (id, name, slug, "createdAt", "updatedAt")
SELECT
    'org_test_123',
    'Test Organization',
    'test-org',
    NOW(),
    NOW()
WHERE NOT EXISTS (
    SELECT 1 FROM "organization" WHERE id = 'org_test_123'
);

-- Insert test project (only if not exists)
INSERT INTO "Project" (id, name, slug, "organizationId", "createdAt", "updatedAt")
SELECT
    'proj_test_123',
    'Test Project',
    'test-project',
    'org_test_123',
    NOW(),
    NOW()
WHERE NOT EXISTS (
    SELECT 1 FROM "Project" WHERE id = 'proj_test_123'
);

-- Note: The above INSERT statements will fail on first run before migrations.
-- That's expected - they'll succeed after `prisma migrate deploy` is run.
