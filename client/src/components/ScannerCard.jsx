const ScannerCard = ({ title, description, children }) => {
  return (
    <div className="card max-w-2xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-800 mb-2">{title}</h2>
        <p className="text-slate-600">{description}</p>
      </div>
      {children}
    </div>
  );
};

export default ScannerCard;
