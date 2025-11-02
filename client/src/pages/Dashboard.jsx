import { Link } from 'react-router-dom';
import { Folder, HardDrive, Share2, ArrowRight, FileText } from 'lucide-react';

const Dashboard = () => {
  const sources = [
    {
      id: 'local',
      title: 'Local Directory',
      description: 'Scan files from your local machine',
      icon: Folder,
      color: 'bg-blue-500',
      route: '/scanner?source=local',
    },
    {
      id: 'drive',
      title: 'Google Drive',
      description: 'Connect and scan Google Drive',
      icon: HardDrive,
      color: 'bg-green-500',
      route: '/scanner?source=drive',
    },
    {
      id: 'sharepoint',
      title: 'SharePoint',
      description: 'Connect and scan SharePoint',
      icon: Share2,
      color: 'bg-primary-500',
      route: '/scanner?source=sharepoint',
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold text-slate-800 mb-2">
          Welcome to RAG File Scanner
        </h1>
        <p className="text-lg text-slate-600">
          Choose a source to scan and extract file structures as JSON
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {sources.map((source) => {
          const Icon = source.icon;
          return (
            <Link
              key={source.id}
              to={source.route}
              className="card group hover:shadow-xl transition-all duration-300 hover:scale-105"
            >
              <div className="flex flex-col items-center text-center">
                <div className={`${source.color} w-16 h-16 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-800 mb-2">
                  {source.title}
                </h3>
                <p className="text-slate-600 mb-4">
                  {source.description}
                </p>
                <div className="flex items-center text-primary-600 font-medium group-hover:translate-x-1 transition-transform">
                  Get Started
                  <ArrowRight className="w-5 h-5 ml-2" />
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      <div className="card bg-gradient-to-br from-primary-50 to-blue-50 border-primary-200">
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0">
            <FileText className="w-8 h-8 text-primary-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-800 mb-2">
              How It Works
            </h3>
            <ol className="list-decimal list-inside space-y-2 text-slate-700">
              <li>Select a file source (Local, Google Drive, or SharePoint)</li>
              <li>Authenticate if required (for cloud services)</li>
              <li>Choose directories to scan</li>
              <li>View extracted file structure in JSON format</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
