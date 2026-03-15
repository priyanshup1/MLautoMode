import { useState } from 'react'
import axios from 'axios'
import { UploadCloud, Play, CheckCircle, AlertTriangle, FileText, FastForward } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function PredictTab({ sessionId, problemType, onRetrainComplete, proMode }) {
    const [file, setFile] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [predictions, setPredictions] = useState(null)
    const [modelVersion, setModelVersion] = useState(null)
    const [driftReport, setDriftReport] = useState(null)

    const [learningLoading, setLearningLoading] = useState(false)

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0])
            setError(null)
            setPredictions(null)
            setDriftReport(null)
        }
    }

    const handlePredict = async () => {
        if (!file) return
        setLoading(true)
        setError(null)

        const formData = new FormData()
        formData.append('file', file)
        if (proMode) formData.append('pro_mode', 'true')

        try {
            const res = await axios.post(`${API_BASE}/api/predict/${sessionId}`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })
            setPredictions(res.data.predictions)
            setModelVersion(res.data.version)
            setDriftReport(res.data.drift_report)
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to generate predictions')
        } finally {
            setLoading(false)
        }
    }

    const handleRetrain = async () => {
        if (!file) return
        setLearningLoading(true)
        setError(null)

        const formData = new FormData()
        formData.append('file', file)

        try {
            const res = await axios.post(`${API_BASE}/api/retrain/${sessionId}`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })
            onRetrainComplete(res.data)
            setPredictions(null)
            setFile(null)
            alert(`Models successfully retrained! Now using version v${res.data.version}`)
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to retrain models')
        } finally {
            setLearningLoading(false)
        }
    }

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-3">
                    <FastForward className="h-6 w-6 text-indigo-600" />
                    Predict & Continual Learning
                </h1>
                <p className="mt-1 text-slate-500">
                    Upload new data to generate predictions using your trained model.
                    If the data includes actual labels, you can use it to retrain and improve the model.
                </p>
            </div>

            {/* Upload Section */}
            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                <div
                    className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors
                    ${file ? 'border-indigo-400 bg-indigo-50/50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}`}
                >
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        className="hidden"
                        id="predict-upload"
                    />
                    <label htmlFor="predict-upload" className="cursor-pointer flex flex-col items-center">
                        {file ? (
                            <>
                                <div className="h-16 w-16 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center mb-4">
                                    <FileText className="h-8 w-8" />
                                </div>
                                <h3 className="text-lg font-semibold text-slate-800">{file.name}</h3>
                                <p className="text-slate-500 mt-1 text-sm">{(file.size / 1024).toFixed(1)} KB</p>
                                <p className="text-indigo-600 font-medium text-sm mt-4 hover:underline">Choose a different file</p>
                            </>
                        ) : (
                            <>
                                <div className="h-16 w-16 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center mb-4">
                                    <UploadCloud className="h-8 w-8" />
                                </div>
                                <h3 className="text-lg font-semibold text-slate-800">Upload Dataset for Prediction</h3>
                                <p className="text-slate-500 mt-1 mb-4 text-sm">CSV files only. Must match or contain the training features.</p>
                                <span className="bg-white border border-slate-200 text-slate-700 px-4 py-2 rounded-full text-sm font-medium shadow-sm">
                                    Browse Files
                                </span>
                            </>
                        )}
                    </label>
                </div>

                {error && (
                    <div className="mt-4 bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
                        <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                        <p className="text-red-700 text-sm">{error}</p>
                    </div>
                )}

                <div className="mt-6 flex gap-4">
                    <button
                        onClick={handlePredict}
                        disabled={!file || loading || learningLoading}
                        className="flex-1 bg-indigo-600 text-white px-6 py-3 rounded-xl font-semibold text-sm hover:bg-indigo-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                        ) : <Play className="h-4 w-4" />}
                        Generate Predictions
                    </button>

                    <button
                        onClick={handleRetrain}
                        disabled={!file || loading || learningLoading}
                        className="flex-1 bg-emerald-600 text-white px-6 py-3 rounded-xl font-semibold text-sm hover:bg-emerald-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
                    >
                        {learningLoading ? (
                            <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                        ) : <CheckCircle className="h-4 w-4" />}
                        Continual Learning (Retrain)
                    </button>
                </div>
                <p className="text-xs text-slate-500 text-center mt-3">
                    Use <strong>Generate Predictions</strong> if the file contains only features.
                    Use <strong>Continual Learning</strong> if the file contains the target variable as well.
                </p>

                {/* Data Drift Warning Display */}
                {driftReport && driftReport.drift_detected && (
                    <div className="mt-6 bg-orange-50 border border-orange-200 rounded-xl p-5 flex items-start gap-3 shadow-sm">
                        <AlertTriangle className="h-6 w-6 text-orange-500 mt-0.5 shrink-0" />
                        <div>
                            <h4 className="text-orange-800 font-bold mb-1">Data Drift Detected!</h4>
                            <p className="text-orange-700 text-sm mb-3">
                                {driftReport.message} The distribution of the following features has significantly shifted compared to the training data:
                            </p>
                            <ul className="list-disc pl-5 text-sm text-orange-700 space-y-1">
                                {driftReport.details.filter(d => d.is_drifted).map(d => (
                                    <li key={d.column}>
                                        <span className="font-semibold">{d.column}</span>
                                        <span className="text-orange-600/80 ml-2">(p-value: {d.p_value})</span>
                                    </li>
                                ))}
                            </ul>
                            <div className="mt-3 text-xs text-orange-600 bg-orange-100 rounded p-2 inline-block">
                                Tip: Upload a file containing actual labels and click 'Continual Learning (Retrain)' to adapt the model to this new data distribution.
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Predictions Table */}
            {predictions && (
                <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                    <div className="px-5 py-4 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <FastForward className="h-5 w-5 text-indigo-500" />
                            <h2 className="font-semibold text-slate-800">Prediction Results</h2>
                        </div>
                        <span className="bg-indigo-100 text-indigo-700 text-xs px-2.5 py-1 rounded-full font-medium">
                            Model v{modelVersion}
                        </span>
                    </div>

                    <div className="overflow-x-auto max-h-[500px]">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-slate-50 border-b border-slate-200 sticky top-0">
                                <tr>
                                    <th className="px-6 py-3 font-medium text-slate-600 w-24">Row ID</th>
                                    <th className="px-6 py-3 font-medium text-slate-600">Predicted Value</th>
                                    {problemType === 'Classification' && (
                                        <th className="px-6 py-3 font-medium text-slate-600">Confidence Score</th>
                                    )}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {predictions.map((p, index) => (
                                    <tr key={index} className="hover:bg-slate-50">
                                        <td className="px-6 py-3 text-slate-500 font-mono">{index + 1}</td>
                                        <td className="px-6 py-3 font-medium text-slate-900">
                                            {typeof p.prediction === 'number' && p.prediction % 1 !== 0
                                                ? p.prediction.toFixed(4)
                                                : p.prediction}
                                        </td>
                                        {problemType === 'Classification' && (
                                            <td className="px-6 py-3">
                                                {p.confidence !== undefined ? (
                                                    <div className="flex items-center gap-2">
                                                        <div className="w-24 bg-slate-200 rounded-full h-2">
                                                            <div
                                                                className={`h-2 rounded-full ${p.confidence > 0.8 ? 'bg-emerald-500' : p.confidence > 0.5 ? 'bg-amber-500' : 'bg-red-500'}`}
                                                                style={{ width: `${p.confidence * 100}%` }}
                                                            ></div>
                                                        </div>
                                                        <span className="text-slate-600 font-mono text-xs">{(p.confidence * 100).toFixed(1)}%</span>
                                                    </div>
                                                ) : <span className="text-slate-400">-</span>}
                                            </td>
                                        )}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    )
}
