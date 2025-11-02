import { useState, useEffect } from 'react';
import { registerMachine } from '../services/api';

const MACHINE_ID_KEY = 'machineId';

/**
 * Custom hook to manage machineId registration and persistence
 * 
 * On first load, if no machineId exists in localStorage, calls the backend
 * to register the machine and get a persistent machineId.
 * The machineId is stored in localStorage and reused on subsequent loads.
 * 
 * @returns {Object} { machineId, isLoading, error }
 */
export const useMachineId = () => {
  const [machineId, setMachineId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initializeMachineId = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Check if machineId exists in localStorage
        const storedMachineId = localStorage.getItem(MACHINE_ID_KEY);

        if (storedMachineId) {
          // Use existing machineId from localStorage
          setMachineId(storedMachineId);
          setIsLoading(false);
          return;
        }

        // No machineId in localStorage - register with backend
        // Backend will use fingerprint (IP + user-agent) to detect if this is a returning client
        const response = await registerMachine();
        const newMachineId = response.machineId;

        // Store machineId in localStorage
        localStorage.setItem(MACHINE_ID_KEY, newMachineId);
        setMachineId(newMachineId);

        console.log(`Machine registered: ${newMachineId} (${response.isNew ? 'new' : 'existing'})`);
      } catch (err) {
        console.error('Failed to register machine:', err);
        setError(err.message || 'Failed to register machine');
      } finally {
        setIsLoading(false);
      }
    };

    initializeMachineId();
  }, []);

  return { machineId, isLoading, error };
};

