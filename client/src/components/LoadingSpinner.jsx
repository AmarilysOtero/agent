const LoadingSpinner = ({ size = 'md', message = 'Loading...' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-4 py-8">
      <div className={`animate-spin rounded-full border-4 border-slate-200 border-t-primary-600 ${sizeClasses[size]}`}></div>
      {message && (
        <p className="text-slate-600 font-medium">{message}</p>
      )}
    </div>
  );
};

export default LoadingSpinner;
