import { useState } from 'react'
import axios from 'axios'
import { GitCommit, CheckCircle2, AlertCircle, Table, Download, ChevronDown } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const SCALERS = [
    { value: 'standard', label: 'Standard Scaler (Z-score)', desc: 'Centers around mean, scales by std deviation. Best general-purpose choice.' },
    { value: 'minmax', label: 'Min-Max Scaler', desc: 'Rescales to [0, 1]. Good for neural networks and image data.' },
    { value: 'robust', label: 'Robust Scaler (IQR-based)', desc: 'Uses median & IQR — less sensitive to outliers than StandardScaler.' },
]

export default function Preprocessing({ sessionId, onComplete, existingData, proMode }) {
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [scaleFeatures, setScaleFeatures] = useState(true)
    const [scalerType, setScalerType] = useState('standard')

    // Pro Mode states
    const [imputationStrategy, setImputationStrategy] = useState('median')
    const [encodingStrategy, setEncodingStrategy] = useState('onehot')
    const [polynomialDegree, setPolynomialDegree] = useState(1)
    const [featureSelectionK, setFeatureSelectionK] = useState(0)
    const [handleOutliers, setHandleOutliers] = useState(false)

    const handlePreprocess = async () => {
        setLoading(true)
        setError(null)
        try {
            const res = await axios.post(
                `${API_BASE}/api/preprocess/${sessionId}`,
                {
                    scale_features: scaleFeatures,
                    scaler_type: scalerType,
                    imputation_strategy: imputationStrategy,
                    encoding_strategy: encodingStrategy,
                    polynomial_degree: polynomialDegree,
                    feature_selection_k: featureSelectionK,
                    handle_outliers: handleOutliers,
                    pro_mode: proMode
                },
                { headers: { 'Content-Type': 'application/json' } }
            )
            onComplete(res.data)
        } catch (err) {
            setError(err.response?.data?.detail || 'Preprocessing failed')
        } finally {
            setLoading(false)
        }
    }

    const handleDownload = () => {
        if (!existingData?.cleaned_csv_b64) return
        const bytes = atob(existingData.cleaned_csv_b64)
        const arr = new Uint8Array(bytes.length)
        for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i)
        const blob = new Blob([arr], { type: 'text/csv' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url; a.download = 'cleaned_dataset.csv'; a.click()
        URL.revokeObjectURL(url)
    }

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-slate-900">Data Preprocessing & Cleaning</h1>
                <p className="mt-1 text-slate-500">Configure and run the automated cleaning pipeline.</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                    { title: 'Handle Missing Values', desc: 'Numeric → median fill. Categorical → mode fill.' },
                    { title: 'Remove Duplicates', desc: 'All duplicate rows are dropped.' },
                    { title: 'Encode Categoricals', desc: 'One-hot encoding applied. Label encoding for target.' },
                    { title: 'Feature Scaling', desc: 'Choose your scaler below for numeric normalization.' },
                ].map(item => (
                    <div key={item.title} className="bg-white border border-slate-200 rounded-xl p-4 flex gap-3">
                        <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0 mt-0.5" />
                        <div>
                            <p className="font-semibold text-slate-800 text-sm">{item.title}</p>
                            <p className="text-slate-500 text-xs mt-1">{item.desc}</p>
                        </div>
                    </div>
                ))}
            </div>

            {/* Scaling Options */}
            <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="font-medium text-slate-800 text-sm">Feature Scaling</p>
                        <p className="text-xs text-slate-500 mt-0.5">Normalizes numeric features for distance-based models</p>
                    </div>
                    <button onClick={() => setScaleFeatures(!scaleFeatures)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${scaleFeatures ? 'bg-indigo-600' : 'bg-slate-300'}`}>
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform shadow ${scaleFeatures ? 'translate-x-6' : 'translate-x-1'}`} />
                    </button>
                </div>

                {scaleFeatures && (
                    <div className="space-y-2">
                        <p className="text-xs font-medium text-slate-600">Choose Scaler</p>
                        {SCALERS.map(s => (
                            <label key={s.value}
                                className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors
                  ${scalerType === s.value ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200 hover:border-indigo-300'}`}>
                                <input type="radio" name="scaler" value={s.value} checked={scalerType === s.value}
                                    onChange={() => setScalerType(s.value)} className="mt-0.5 accent-indigo-600" />
                                <div>
                                    <p className="text-sm font-medium text-slate-800">{s.label}</p>
                                    <p className="text-xs text-slate-500 mt-0.5">{s.desc}</p>
                                </div>
                            </label>
                        ))}
                    </div>
                )}
                {/* Pro Mode: Advanced Options */}
                {proMode && (
                    <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-5 space-y-6">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="bg-indigo-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">PRO</span>
                            <h3 className="font-bold text-slate-800 text-sm">Advanced Engineering Options</h3>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Imputation Strategy */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">Imputation Strategy</label>
                                <select
                                    value={imputationStrategy}
                                    onChange={(e) => setImputationStrategy(e.target.value)}
                                    className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                                >
                                    <option value="mean">Mean (Average)</option>
                                    <option value="median">Median (Middle value)</option>
                                    <option value="most_frequent">Most Frequent (Mode)</option>
                                    <option value="constant">Constant (0)</option>
                                </select>
                                <p className="text-[10px] text-slate-500 italic">How to handle missing values in numeric columns.</p>
                            </div>

                            {/* Polynomial Features */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">Feature interactions (Polynomial)</label>
                                <select
                                    value={polynomialDegree}
                                    onChange={(e) => setPolynomialDegree(parseInt(e.target.value))}
                                    className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                                >
                                    <option value="1">None (Default)</option>
                                    <option value="2">Degree 2 (Adds quadratic terms &#x26; interactions)</option>
                                </select>
                                <p className="text-[10px] text-slate-500 italic">Captures non-linear relationships. Can explode feature count!</p>
                            </div>

                            {/* Feature Selection */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">Automated Feature Selection (K-Best)</label>
                                <div className="flex items-center gap-3">
                                    <input
                                        type="number"
                                        min="0"
                                        value={featureSelectionK}
                                        onChange={(e) => setFeatureSelectionK(parseInt(e.target.value))}
                                        className="w-20 bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                                    />
                                    <span className="text-sm text-slate-600">Top features (0 = all)</span>
                                </div>
                                <p className="text-[10px] text-slate-500 italic">Reduces noise by keeping only the most statistically significant features.</p>
                            </div>

                            {/* Encoding Strategy */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">Categorical Encoding</label>
                                <select
                                    value={encodingStrategy}
                                    onChange={(e) => setEncodingStrategy(e.target.value)}
                                    className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                                >
                                    <option value="onehot">One-Hot Encoding (Default)</option>
                                    <option value="target">Target Encoding (Pro)</option>
                                </select>
                                <p className="text-[10px] text-slate-500 italic">Target encoding is better for high-cardinality features like 'City' or 'Zipcode'.</p>
                            </div>

                            {/* Outlier Removal */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">Remove Outliers (Isolation Forest)</label>
                                    <button onClick={() => setHandleOutliers(!handleOutliers)}
                                        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${handleOutliers ? 'bg-indigo-600' : 'bg-slate-300'}`}>
                                        <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform shadow ${handleOutliers ? 'translate-x-5' : 'translate-x-1'}`} />
                                    </button>
                                </div>
                                <p className="text-[10px] text-slate-500 italic mt-1.5">Automatically removes anomalous data points using Isolation Forest.</p>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                    <p className="text-red-700 text-sm">{error}</p>
                </div>
            )}

            <button onClick={handlePreprocess} disabled={loading}
                className="w-full bg-indigo-600 text-white px-6 py-3 rounded-xl font-semibold text-sm hover:bg-indigo-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2">
                {loading
                    ? <><span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" /> Cleaning Data...</>
                    : <><GitCommit className="h-4 w-4" /> Run Preprocessing</>}
            </button>

            {existingData && (
                <div className="space-y-5">
                    <div className="bg-green-50 border border-green-200 rounded-xl p-4">
                        <p className="font-semibold text-green-800">{existingData.message}</p>
                        <div className="grid grid-cols-3 gap-4 mt-3">
                            <StatCard label="Duplicates Removed" value={existingData.stats.removed_duplicates} />
                            <StatCard label="Original Rows" value={existingData.stats.original_rows} />
                            <StatCard label="Imputed Columns" value={existingData.stats.imputed_columns?.length || 0} />
                        </div>
                    </div>

                    {/* Step-by-step log */}
                    {existingData.step_log?.length > 0 && (
                        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                            <div className="px-5 py-3 bg-slate-50 border-b text-sm font-medium text-slate-700">🪜 Preprocessing Steps Applied</div>
                            <div className="divide-y divide-slate-100">
                                {existingData.step_log.map((s, i) => (
                                    <div key={i} className="px-5 py-3 flex gap-3 text-sm">
                                        <span className="h-5 w-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs font-bold shrink-0">{i + 1}</span>
                                        <div>
                                            <span className="font-medium text-slate-800">{s.step}:</span>
                                            <span className="text-slate-600 ml-1">{s.detail}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Preview table */}
                    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-100 bg-slate-50">
                            <div className="flex items-center gap-2">
                                <Table className="h-4 w-4 text-slate-500" />
                                <span className="text-sm font-medium text-slate-700">Cleaned Data Preview (first 10 rows)</span>
                            </div>
                            <button onClick={handleDownload}
                                className="flex items-center gap-1.5 text-xs text-indigo-600 hover:text-indigo-800 font-medium border border-indigo-200 px-3 py-1.5 rounded-lg hover:bg-indigo-50 transition-colors">
                                <Download className="h-3.5 w-3.5" /> Download CSV
                            </button>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="text-xs w-full">
                                <thead className="bg-slate-50 border-b border-slate-200">
                                    <tr>
                                        {Object.keys(existingData.preview[0] || {}).slice(0, 8).map(col => (
                                            <th key={col} className="px-3 py-2 text-left font-medium text-slate-600 truncate max-w-[120px]">{col}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {existingData.preview.map((row, i) => (
                                        <tr key={i} className="hover:bg-slate-50">
                                            {Object.values(row).slice(0, 8).map((val, j) => (
                                                <td key={j} className="px-3 py-2 text-slate-700 truncate max-w-[120px]">
                                                    {typeof val === 'number' ? val.toFixed(4) : String(val)}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

function StatCard({ label, value }) {
    return (
        <div className="bg-white rounded-lg p-3 text-center border border-green-100">
            <p className="text-lg font-bold text-slate-800">{value}</p>
            <p className="text-xs text-slate-500 mt-0.5">{label}</p>
        </div>
    )
}
