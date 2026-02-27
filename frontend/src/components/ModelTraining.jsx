import { useState } from 'react'
import axios from 'axios'
import { Play, Trophy, AlertCircle, TrendingUp } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

const CLASSIFICATION_MODELS = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN']
const REGRESSION_MODELS = ['Linear Regression', 'Random Forest Regressor', 'Decision Tree', 'SVR']

function MetricBadge({ value, isPercent }) {
    const display = isPercent ? `${(value * 100).toFixed(1)}%` : typeof value === 'number' ? value.toFixed(4) : value
    return <span className="font-mono text-sm text-slate-800">{display}</span>
}

export default function ModelTraining({ sessionId, problemType, onComplete, existingData }) {
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [selectedModels, setSelectedModels] = useState(
        problemType === 'Classification' ? CLASSIFICATION_MODELS : REGRESSION_MODELS
    )

    const allModels = problemType === 'Classification' ? CLASSIFICATION_MODELS : REGRESSION_MODELS

    const toggleModel = (m) => {
        setSelectedModels(prev =>
            prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]
        )
    }

    const handleTrain = async () => {
        if (selectedModels.length === 0) return
        setLoading(true)
        setError(null)
        try {
            const res = await axios.post(
                `${API_BASE}/api/train/${sessionId}`,
                { models: selectedModels },
                { headers: { 'Content-Type': 'application/json' } }
            )
            onComplete(res.data)
        } catch (err) {
            setError(err.response?.data?.detail || 'Training failed')
        } finally {
            setLoading(false)
        }
    }

    const metrics = problemType === 'Classification'
        ? [{ key: 'Accuracy', pct: true }, { key: 'Precision', pct: true }, { key: 'Recall', pct: true }, { key: 'F1 Score', pct: true }]
        : [{ key: 'R2 Score', pct: false }, { key: 'MAE', pct: false }, { key: 'MSE', pct: false }, { key: 'RMSE', pct: false }]

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-slate-900">Model Training Panel</h1>
                <p className="mt-1 text-slate-500">Select models to train and compare. Problem type: <strong className="text-indigo-600">{problemType}</strong></p>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between">
                    <p className="font-semibold text-slate-800 text-sm">Select Models to Train</p>
                    <button onClick={() => setSelectedModels(allModels)} className="text-xs text-indigo-600 hover:underline">Select All</button>
                </div>
                <div className="flex flex-wrap gap-2">
                    {allModels.map(m => (
                        <button
                            key={m}
                            onClick={() => toggleModel(m)}
                            className={`px-4 py-2 rounded-full text-sm font-medium border transition-colors
                ${selectedModels.includes(m)
                                    ? 'bg-indigo-600 border-indigo-600 text-white'
                                    : 'bg-white border-slate-300 text-slate-600 hover:border-indigo-400'}`}
                        >
                            {m}
                        </button>
                    ))}
                </div>
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                    <p className="text-red-700 text-sm">{error}</p>
                </div>
            )}

            <button
                onClick={handleTrain}
                disabled={loading || selectedModels.length === 0}
                className="w-full bg-indigo-600 text-white px-6 py-3 rounded-xl font-semibold text-sm hover:bg-indigo-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
                {loading ? (
                    <>
                        <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                        Training {selectedModels.length} model{selectedModels.length > 1 ? 's' : ''}...
                    </>
                ) : (
                    <>
                        <Play className="h-4 w-4" />
                        Train {selectedModels.length} Model{selectedModels.length > 1 ? 's' : ''}
                    </>
                )}
            </button>

            {existingData && existingData.results && (
                <div className="space-y-4">
                    <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
                        <Trophy className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
                        <div>
                            <p className="font-semibold text-amber-800">Best Model: {existingData.results[0]?.model_name}</p>
                            <p className="text-sm text-amber-700 mt-0.5">
                                {problemType === 'Classification'
                                    ? `Accuracy: ${(existingData.results[0].metrics['Accuracy'] * 100).toFixed(1)}%`
                                    : `R² Score: ${existingData.results[0].metrics['R2 Score'].toFixed(4)}`}
                            </p>
                        </div>
                    </div>

                    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                        <div className="px-5 py-3 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
                            <TrendingUp className="h-4 w-4 text-slate-500" />
                            <span className="text-sm font-medium text-slate-700">Model Comparison</span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead className="bg-slate-50 border-b border-slate-200">
                                    <tr>
                                        <th className="px-5 py-3 text-left font-medium text-slate-600">Model</th>
                                        {metrics.map(m => (
                                            <th key={m.key} className="px-5 py-3 text-left font-medium text-slate-600">{m.key}</th>
                                        ))}
                                        <th className="px-5 py-3 text-left font-medium text-slate-600">Rank</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {existingData.results.map((r, i) => (
                                        <tr key={r.model_name} className={i === 0 ? 'bg-amber-50' : 'hover:bg-slate-50'}>
                                            <td className="px-5 py-3 font-medium text-slate-800">
                                                {i === 0 && <Trophy className="inline h-3.5 w-3.5 text-amber-500 mr-1.5" />}
                                                {r.model_name}
                                            </td>
                                            {metrics.map(m => (
                                                <td key={m.key} className="px-5 py-3">
                                                    <MetricBadge value={r.metrics[m.key]} isPercent={m.pct} />
                                                </td>
                                            ))}
                                            <td className="px-5 py-3">
                                                <span className={`text-xs px-2 py-1 rounded-full font-medium ${i === 0 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'}`}>
                                                    #{i + 1}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 space-y-3">
                        <p className="font-semibold text-slate-800 text-sm">📚 Why These Metrics Matter</p>
                        {problemType === 'Classification' ? (
                            <ul className="space-y-2 text-sm text-slate-600">
                                <li><strong>Accuracy</strong> — Overall % of correct predictions. Good baseline, but can be misleading on imbalanced data.</li>
                                <li><strong>Precision</strong> — Of all positive predictions, how many were correct? Reduces false positives.</li>
                                <li><strong>Recall</strong> — Of all actual positives, how many did we catch? Critical in medical/fraud contexts.</li>
                                <li><strong>F1 Score</strong> — Harmonic mean of Precision & Recall. Best when classes are imbalanced.</li>
                            </ul>
                        ) : (
                            <ul className="space-y-2 text-sm text-slate-600">
                                <li><strong>R² Score</strong> — Proportion of variance explained. Closer to 1 is better. Main performance indicator.</li>
                                <li><strong>MAE</strong> — Mean Absolute Error. Average magnitude of error in original units. Easy to interpret.</li>
                                <li><strong>RMSE</strong> — Root Mean Squared Error. Penalizes large errors more heavily than MAE.</li>
                                <li><strong>MSE</strong> — Mean Squared Error. Used internally; RMSE is its square root for interpretability.</li>
                            </ul>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
