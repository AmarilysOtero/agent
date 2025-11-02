import { Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Scanner from './pages/Scanner';
import Results from './pages/Results';
import Retrieve from './pages/Retrieve';
import { useMachineId } from './hooks/useMachineId';

function App() {
  // Initialize machineId on app load - this will register the machine if needed
  const { machineId, isLoading, error } = useMachineId();

  // Log machine registration status (optional, for debugging)
  useEffect(() => {
    if (machineId && !isLoading) {
      console.log(`App initialized with machineId: ${machineId}`);
    }
    if (error) {
      console.error('Machine registration error:', error);
    }
  }, [machineId, isLoading, error]);

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/scanner" element={<Scanner />} />
        <Route path="/results" element={<Results />} />
        <Route path="/retrieve" element={<Retrieve />} />
      </Routes>
    </Layout>
  );
}

export default App;
