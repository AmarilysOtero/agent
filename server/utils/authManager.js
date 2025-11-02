/**
 * Authentication state manager
 * In production, use proper session management (Redis, database, etc.)
 */

const authState = new Map();

// Store auth state
const storeAuth = (stateId, authData) => {
  authState.set(stateId, {
    ...authData,
    createdAt: Date.now()
  });
};

// Get auth state
const getAuth = (stateId) => {
  return authState.get(stateId);
};

// Check if auth is valid (not expired)
const isAuthValid = (stateId, maxAge = 3600000) => { // 1 hour default
  const authData = authState.get(stateId);
  if (!authData) return false;
  
  const age = Date.now() - authData.createdAt;
  return age < maxAge;
};

// Remove auth state
const removeAuth = (stateId) => {
  authState.delete(stateId);
};

// Clean up expired auth states
const cleanupExpired = (maxAge = 3600000) => {
  const now = Date.now();
  for (const [stateId, authData] of authState.entries()) {
    if (now - authData.createdAt > maxAge) {
      authState.delete(stateId);
    }
  }
};

// Run cleanup every 30 minutes
setInterval(() => cleanupExpired(), 30 * 60 * 1000);

module.exports = {
  storeAuth,
  getAuth,
  isAuthValid,
  removeAuth,
  authState // Export for access in routes
};
