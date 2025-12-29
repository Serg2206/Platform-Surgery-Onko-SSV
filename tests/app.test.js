/**
 * Tests for Surgery-Onko-SSV Platform
 */

const request = require('supertest');
const app = require('../src/index');

describe('Platform Surgery-Onko-SSV API Tests', () => {
  
  describe('GET /health', () => {
    it('should return 200 and health status', async () => {
      const response = await request(app).get('/health');
      
      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status', 'OK');
      expect(response.body).toHaveProperty('service', 'Surgery-Onko-SSV Platform');
      expect(response.body).toHaveProperty('version', '1.0.0');
      expect(response.body).toHaveProperty('timestamp');
    });
  });

  describe('GET /api/v1/info', () => {
    it('should return platform information', async () => {
      const response = await request(app).get('/api/v1/info');
      
      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('name', 'Platform-Surgery-Onko-SSV');
      expect(response.body).toHaveProperty('description');
      expect(response.body).toHaveProperty('features');
      expect(response.body).toHaveProperty('technologies');
      
      expect(response.body.features).toBeInstanceOf(Array);
      expect(response.body.features.length).toBeGreaterThan(0);
      
      expect(response.body.technologies).toContain('TensorFlow.js');
      expect(response.body.technologies).toContain('Express');
    });
  });

  describe('GET /nonexistent', () => {
    it('should return 404 for non-existent routes', async () => {
      const response = await request(app).get('/nonexistent');
      
      expect(response.status).toBe(404);
      expect(response.body).toHaveProperty('error', 'Not Found');
    });
  });

});
