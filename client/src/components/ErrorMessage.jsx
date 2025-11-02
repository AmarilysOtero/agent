import { AlertCircle } from 'lucide-react';

const ErrorMessage = ({ message, onDismiss }) => {
  return (
    <div className="card bg-red-50 border-red-200">
      <div className="flex items-start space-x-3">
        <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-red-800 font-medium">Error</p>
          <p className="text-red-700">{message}</p>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-red-600 hover:text-red-800 transition-colors"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;
