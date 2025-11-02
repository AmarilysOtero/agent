import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { useForm } from 'react-hook-form';
import {
  setSource,
  setAuthState,
  setScanResults,
  setLoading,
  setError,
} from '../store/slices/scannerSlice';
import { scanLocal, authenticateGoogleDrive, scanGoogleDrive, authenticateSharePoint, storeInNeo4j } from '../services/api';
import { Folder, HardDrive, Share2, Scan, Check } from 'lucide-react';
import ScannerCard from '../components/ScannerCard';
import { store } from '../store/store';

const Scanner = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  
  const { isLoading, error, scanResults } = useSelector((state) => state.scanner);
  const [source, setSourceType] = useState(searchParams.get('source') || 'local');
  const [authUrl, setAuthUrl] = useState(null);
  
  useEffect(() => {
    dispatch(setSource(source));
  }, [source, dispatch]);

  const handleLocalScan = async (data) => {
    try {
      dispatch(setLoading(true));
      dispatch(setError(null));
      console.log('Starting scan for:', data.directoryPath);
      
      // Scan the directory
      const result = await scanLocal(data.directoryPath);
      console.log('Scan result received:', result);
      
      dispatch(setScanResults(result));
      navigate('/results');
    } catch (err) {
      console.error('Scan error:', err);
      dispatch(setError(err.response?.data?.error || 'Failed to scan local directory'));
    } finally {
      dispatch(setLoading(false));
    }
  };

  const handleGoogleDriveAuth = async () => {
    try {
      const result = await authenticateGoogleDrive({});
      setAuthUrl(result.authUrl);
      dispatch(setAuthState(result.stateId));
      window.open(result.authUrl, '_blank', 'width=600,height=700');
    } catch (err) {
      dispatch(setError(err.response?.data?.error || 'Failed to authenticate with Google Drive'));
    }
  };

  const handleSharePointAuth = async (data) => {
    try {
      const result = await authenticateSharePoint(data);
      setAuthUrl(result.authUrl);
      dispatch(setAuthState(result.stateId));
      window.open(result.authUrl, '_blank', 'width=600,height=700');
    } catch (err) {
      dispatch(setError(err.response?.data?.error || 'Failed to authenticate with SharePoint'));
    }
  };

  const handleDriveScan = async () => {
    try {
      dispatch(setLoading(true));
      dispatch(setError(null));
      const state = store.getState();
      const stateId = state?.scanner?.authStateId;
      if (!stateId) {
        dispatch(setError('Please authenticate first'));
        dispatch(setLoading(false));
        return;
      }
      const result = await scanGoogleDrive(stateId);
      dispatch(setScanResults(result));
      navigate('/results');
    } catch (err) {
      dispatch(setError(err.response?.data?.error || 'Failed to scan Google Drive'));
    } finally {
      dispatch(setLoading(false));
    }
  };

  const renderContent = () => {
    switch (source) {
      case 'local':
        return <LocalScanner onSubmit={handleLocalScan} />;
      case 'drive':
        return <GoogleDriveScanner onAuth={handleGoogleDriveAuth} onScan={handleDriveScan} />;
      case 'sharepoint':
        return <SharePointScanner onAuth={handleSharePointAuth} />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-4 mb-6">
        {[
          { id: 'local', icon: Folder, label: 'Local' },
          { id: 'drive', icon: HardDrive, label: 'Google Drive' },
          { id: 'sharepoint', icon: Share2, label: 'SharePoint' }
        ].map((src) => {
          const Icon = src.icon;
          return (
            <button
              key={src.id}
              onClick={() => setSourceType(src.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                source === src.id
                  ? 'bg-primary-600 text-white'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{src.label}</span>
            </button>
          );
        })}
      </div>

      {error && (
        <div className="card bg-red-50 border-red-200">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {renderContent()}
    </div>
  );
};

const LocalScanner = ({ onSubmit }) => {
  const { register, handleSubmit, formState: { errors } } = useForm();
  
  return (
    <ScannerCard
      title="Scan Local Directory"
      description="Enter the path to the directory you want to scan"
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Directory Path
          </label>
          <input
            type="text"
            className="input"
            placeholder="C:\Users\Username\Documents or /home/user/documents"
            {...register('directoryPath', { required: 'Directory path is required' })}
          />
          {errors.directoryPath && (
            <p className="text-red-600 text-sm mt-1">{errors.directoryPath.message}</p>
          )}
        </div>
        <button type="submit" className="btn btn-primary w-full flex items-center justify-center">
          <Scan className="w-5 h-5 mr-2" />
          Scan Directory
        </button>
      </form>
    </ScannerCard>
  );
};

const GoogleDriveScanner = ({ onAuth, onScan }) => {
  const { authStateId, isLoading } = useSelector((state) => state.scanner);
  
  return (
    <ScannerCard
      title="Scan Google Drive"
      description="Authenticate with Google Drive to scan your files"
    >
      {!authStateId ? (
        <div className="space-y-4">
          <p className="text-slate-600">
            Click the button below to authenticate with Google Drive. You'll be redirected to Google's authentication page.
          </p>
          <button onClick={onAuth} className="btn btn-primary w-full">
            Authenticate with Google Drive
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center space-x-2 text-green-600 bg-green-50 rounded-lg p-3">
            <Check className="w-5 h-5" />
            <span>Authenticated successfully!</span>
          </div>
          <button 
            onClick={onScan} 
            disabled={isLoading}
            className="btn btn-primary w-full flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Scanning...
              </>
            ) : (
              <>
                <Scan className="w-5 h-5 mr-2" />
                Scan Google Drive
              </>
            )}
          </button>
        </div>
      )}
    </ScannerCard>
  );
};

const SharePointScanner = ({ onAuth }) => {
  const { register, handleSubmit, formState: { errors } } = useForm();
  
  return (
    <ScannerCard
      title="Scan SharePoint"
      description="Enter your SharePoint credentials to authenticate"
    >
      <form onSubmit={handleSubmit(onAuth)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            SharePoint Site URL
          </label>
          <input
            type="text"
            className="input"
            placeholder="https://yourtenant.sharepoint.com"
            {...register('siteUrl', { required: 'Site URL is required' })}
          />
          {errors.siteUrl && (
            <p className="text-red-600 text-sm mt-1">{errors.siteUrl.message}</p>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Client ID
          </label>
          <input
            type="text"
            className="input"
            placeholder="Your application client ID"
            {...register('clientId', { required: 'Client ID is required' })}
          />
          {errors.clientId && (
            <p className="text-red-600 text-sm mt-1">{errors.clientId.message}</p>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Tenant ID
          </label>
          <input
            type="text"
            className="input"
            placeholder="Your Azure AD tenant ID"
            {...register('tenantId', { required: 'Tenant ID is required' })}
          />
          {errors.tenantId && (
            <p className="text-red-600 text-sm mt-1">{errors.tenantId.message}</p>
          )}
        </div>
        <button type="submit" className="btn btn-primary w-full">
          Authenticate and Scan SharePoint
        </button>
      </form>
    </ScannerCard>
  );
};

export default Scanner;
