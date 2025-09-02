-- Migration: Remove authentication fields from users table
-- Description: Rollback authentication fields for users

-- Remove constraints
ALTER TABLE users DROP CONSTRAINT IF EXISTS check_password_hash_format;
ALTER TABLE users DROP CONSTRAINT IF EXISTS check_valid_roles;
ALTER TABLE users DROP CONSTRAINT IF EXISTS check_valid_permissions;

-- Remove indexes
DROP INDEX IF EXISTS idx_users_verified;
DROP INDEX IF EXISTS idx_users_roles;
DROP INDEX IF EXISTS idx_users_permissions;

-- Remove columns
ALTER TABLE users 
DROP COLUMN IF EXISTS password_hash,
DROP COLUMN IF EXISTS is_verified,
DROP COLUMN IF EXISTS roles,
DROP COLUMN IF EXISTS permissions;

-- Recreate the original trigger
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
