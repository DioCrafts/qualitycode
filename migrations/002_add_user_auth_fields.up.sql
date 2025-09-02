-- Migration: Add authentication fields to users table
-- Description: Add password_hash, verification, roles and permissions to users

-- Add new columns to users table
ALTER TABLE users 
ADD COLUMN password_hash VARCHAR(255),
ADD COLUMN is_verified BOOLEAN DEFAULT FALSE NOT NULL,
ADD COLUMN roles TEXT[] DEFAULT '{}' NOT NULL,
ADD COLUMN permissions TEXT[] DEFAULT '{}' NOT NULL;

-- Make password_hash NOT NULL after adding it (for existing users, set a placeholder)
UPDATE users SET password_hash = 'placeholder_hash_needs_reset' WHERE password_hash IS NULL;
ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL;

-- Add new indexes
CREATE INDEX idx_users_verified ON users(is_verified);
CREATE INDEX idx_users_roles ON users USING GIN(roles);
CREATE INDEX idx_users_permissions ON users USING GIN(permissions);

-- Update the trigger to include new fields
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add constraint to check password hash format (bcrypt starts with $2b$)
ALTER TABLE users ADD CONSTRAINT check_password_hash_format 
CHECK (password_hash ~ '^\$2[aby]\$[0-9]{2}\$[./A-Za-z0-9]{53}$' OR password_hash = 'placeholder_hash_needs_reset');

-- Add constraint for valid roles
ALTER TABLE users ADD CONSTRAINT check_valid_roles
CHECK (
    roles <@ ARRAY['super_admin', 'organization_admin', 'project_maintainer', 'developer', 'viewer']
);

-- Add constraint for valid permissions (basic permission format check)
ALTER TABLE users ADD CONSTRAINT check_valid_permissions
CHECK (
    permissions <@ ARRAY[
        'system:admin',
        'organization:create', 'organization:read', 'organization:update', 'organization:delete', 'organization:manage_users',
        'project:create', 'project:read', 'project:update', 'project:delete', 'project:analyze', 'project:configure',
        'analysis:create', 'analysis:read', 'analysis:delete',
        'user:create', 'user:read', 'user:update', 'user:delete'
    ]
);

-- Create default admin user if no users exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM users LIMIT 1) THEN
        INSERT INTO users (
            email, 
            username, 
            full_name, 
            password_hash, 
            is_active, 
            is_verified, 
            is_admin, 
            roles,
            permissions,
            created_at, 
            updated_at
        ) VALUES (
            'admin@codeant.dev',
            'admin',
            'System Administrator',
            'placeholder_hash_needs_reset',
            TRUE,
            TRUE,
            TRUE,
            ARRAY['super_admin'],
            ARRAY['system:admin'],
            NOW(),
            NOW()
        );
        
        RAISE NOTICE 'Created default admin user: admin@codeant.dev (password needs to be set)';
    END IF;
END $$;
