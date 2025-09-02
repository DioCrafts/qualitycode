import type { Project } from '$lib/types/dashboard';
import type { PageLoad } from './$types';

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
}

export interface PageData {
  projects: Project[];
  organizationId: string;
  user: User;
  error?: string;
}

export const load: PageLoad = async ({ fetch, url }) => {
  // Mock data for development
  // In production, this would fetch from the actual API

  const mockProjects: Project[] = [
    {
      id: 'project-1',
      name: 'CodeAnt Backend',
      language: 'Python',
      lastAnalysis: new Date(),
      metrics: {
        qualityScore: 85.5,
        technicalDebt: 150,
        coverage: 78,
        issues: {
          total: 45,
          critical: 3,
          high: 12
        },
        codeSize: {
          lines: 25000,
          files: 180,
          functions: 450
        }
      },
      status: 'healthy'
    },
    {
      id: 'project-2',
      name: 'CodeAnt Dashboard',
      language: 'TypeScript',
      lastAnalysis: new Date(),
      metrics: {
        qualityScore: 92.0,
        technicalDebt: 80,
        coverage: 85,
        issues: {
          total: 20,
          critical: 0,
          high: 5
        },
        codeSize: {
          lines: 15000,
          files: 120,
          functions: 300
        }
      },
      status: 'healthy'
    }
  ];

  const mockUser: User = {
    id: 'user-1',
    name: 'Developer',
    email: 'dev@codeant.com',
    role: 'developer'
  };

  try {
    // In production, replace with actual API calls
    // const projectsResponse = await fetch('/api/projects');
    // const projects = await projectsResponse.json();

    return {
      projects: mockProjects,
      organizationId: 'org-1',
      user: mockUser
    };
  } catch (error) {
    console.error('Failed to load dashboard data:', error);
    return {
      projects: [],
      organizationId: 'org-1',
      user: mockUser,
      error: 'Failed to load dashboard data'
    };
  }
};