import { CheckCircle } from 'lucide-react';

const SuccessMessage = ({ message, onDismiss }) => {
  return (
    <div className="card bg-green-50 border-green-200">
      <div className="flex items-start space-x-3">
        <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-green-800 font-medium">Success</p>
          <p className="text-green-700">{message}</p>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-green-600 hover:text-green-800 transition-colors"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
};

export default SuccessMessage;
