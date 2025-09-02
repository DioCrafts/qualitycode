-- Rollback de la migración inicial: Esquema base del sistema CodeAnt
-- Fecha: 2024-01-01
-- Descripción: Eliminación de todas las tablas y objetos creados

-- Eliminar triggers
DROP TRIGGER IF EXISTS update_project_rule_configs_updated_at ON project_rule_configs;
DROP TRIGGER IF EXISTS update_rules_updated_at ON rules;
DROP TRIGGER IF EXISTS update_analysis_jobs_updated_at ON analysis_jobs;
DROP TRIGGER IF EXISTS update_file_index_updated_at ON file_index;
DROP TRIGGER IF EXISTS update_repositories_updated_at ON repositories;
DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS update_organizations_updated_at ON organizations;

-- Eliminar función de trigger
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Eliminar tablas en orden inverso (por dependencias)
DROP TABLE IF EXISTS project_rule_configs CASCADE;
DROP TABLE IF EXISTS rules CASCADE;
DROP TABLE IF EXISTS code_metrics CASCADE;
DROP TABLE IF EXISTS analysis_results CASCADE;
DROP TABLE IF EXISTS analysis_jobs CASCADE;
DROP TABLE IF EXISTS file_index CASCADE;
DROP TABLE IF EXISTS repository_branches CASCADE;
DROP TABLE IF EXISTS repositories CASCADE;
DROP TABLE IF EXISTS projects CASCADE;
DROP TABLE IF EXISTS organization_members CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS organizations CASCADE;

-- Eliminar extensiones
DROP EXTENSION IF EXISTS "pg_trgm";
DROP EXTENSION IF EXISTS "uuid-ossp";
