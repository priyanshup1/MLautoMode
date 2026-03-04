import { useState } from 'react'
import { Upload, Activity, GitCommit, Play, BarChart2, Lightbulb } from 'lucide-react'
import DatasetUpload from './components/DatasetUpload'
import DataAnalysis from './components/DataAnalysis'
import Preprocessing from './components/Preprocessing'
import ModelTraining from './components/ModelTraining'
import Visualizations from './components/Visualizations'
import PredictTab from './components/PredictTab'

function App() {
  const [activeTab, setActiveTab] = useState('upload')
  const [sessionId, setSessionId] = useState(null)
  const [problemType, setProblemType] = useState(null)
  const [proMode, setProMode] = useState(false)

  const [uploadData, setUploadData] = useState(null)
  const [analysisData, setAnalysisData] = useState(null)
  const [preprocessData, setPreprocessData] = useState(null)
  const [trainingData, setTrainingData] = useState(null)

  const tabs = [
    { id: 'upload', icon: Upload, label: 'Upload Dataset', disabled: false },
    { id: 'analysis', icon: Activity, label: 'Data Analysis', disabled: !sessionId },
    { id: 'preprocess', icon: GitCommit, label: 'Preprocessing', disabled: !sessionId },
    { id: 'train', icon: Play, label: 'Train Models', disabled: !preprocessData },
    { id: 'visualize', icon: BarChart2, label: 'Visualizations', disabled: !trainingData },
    { id: 'predict', icon: Activity, label: 'Predict', disabled: !trainingData },
  ]

  const handleUploadComplete = (data) => {
    setSessionId(data.session_id)
    setProblemType(data.analysis.problem_type)
    setUploadData(data)
    setAnalysisData(null)
    setPreprocessData(null)
    setTrainingData(null)
    setActiveTab('analysis')
  }

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data)
    setActiveTab('preprocess')
  }

  const handlePreprocessComplete = (data) => {
    setPreprocessData(data)
    setActiveTab('train')
  }

  const handleTrainingComplete = (data) => {
    setTrainingData(data)
    setActiveTab('visualize')
  }

  const handleRetrainComplete = (data) => {
    // When retraining completes, we update the training data with new model results
    setTrainingData(data)
    // Optional: could automatically navigate back to visualize if desired 
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Navbar */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Lightbulb className="h-8 w-8 text-indigo-600" />
              <span className="ml-2 text-xl font-bold text-slate-900">Mini AutoML</span>
              <span className="ml-2 text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-medium">V2</span>
            </div>
            <div className="flex space-x-1 sm:space-x-2 items-center overflow-x-auto">
              {tabs.map((tab) => {
                const Icon = tab.icon
                const isDone =
                  (tab.id === 'upload' && uploadData) ||
                  (tab.id === 'analysis' && analysisData) ||
                  (tab.id === 'preprocess' && preprocessData) ||
                  (tab.id === 'train' && trainingData)
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    disabled={tab.disabled}
                    className={`flex items-center px-3 py-2 text-xs sm:text-sm font-medium rounded-md transition-colors whitespace-nowrap gap-1.5
                      ${activeTab === tab.id
                        ? 'bg-indigo-50 text-indigo-700'
                        : tab.disabled
                          ? 'text-slate-400 cursor-not-allowed'
                          : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'}`}
                  >
                    <Icon className="h-4 w-4" />
                    {tab.label}
                    {isDone && <span className="h-1.5 w-1.5 rounded-full bg-green-500" />}
                  </button>
                )
              })}
            </div>
            {/* Pro Mode Toggle */}
            <div className="flex items-center space-x-2 ml-4 border-l border-slate-200 pl-4 py-2 hidden sm:flex">
              <span className={`text-xs font-bold leading-none ${proMode ? 'text-indigo-600' : 'text-slate-400'}`}>PRO</span>
              <button
                type="button"
                onClick={() => setProMode(!proMode)}
                className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${proMode ? 'bg-indigo-600' : 'bg-slate-200'}`}
                role="switch"
                aria-checked={proMode}
              >
                <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${proMode ? 'translate-x-4' : 'translate-x-0'}`} />
              </button>
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="h-0.5 bg-slate-100">
          <div
            className="h-full bg-indigo-500 transition-all duration-500"
            style={{ width: `${([uploadData, analysisData, preprocessData, trainingData].filter(Boolean).length / 4) * 100}%` }}
          />
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'upload' && (
          <DatasetUpload onComplete={handleUploadComplete} existingData={uploadData} />
        )}
        {activeTab === 'analysis' && (
          <DataAnalysis sessionId={sessionId} onComplete={handleAnalysisComplete} existingData={analysisData} />
        )}
        {activeTab === 'preprocess' && (
          <Preprocessing sessionId={sessionId} onComplete={handlePreprocessComplete} existingData={preprocessData} proMode={proMode} />
        )}
        {activeTab === 'train' && (
          <ModelTraining sessionId={sessionId} problemType={problemType} onComplete={handleTrainingComplete} existingData={trainingData} proMode={proMode} />
        )}
        {activeTab === 'visualize' && (
          <Visualizations data={trainingData} proMode={proMode} />
        )}
        {activeTab === 'predict' && (
          <PredictTab sessionId={sessionId} problemType={problemType} onRetrainComplete={handleRetrainComplete} />
        )}
      </main>
    </div>
  )
}

export default App
