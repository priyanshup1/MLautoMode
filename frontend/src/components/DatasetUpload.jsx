import { useState, useRef } from 'react'
import axios from 'axios'
import { Upload, FileText, AlertCircle, CheckCircle2, ChevronDown } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function DatasetUpload({ onComplete, existingData }) {
    const [file, setFile] = useState(null)
    const [columns, setColumns] = useState([])
    const [targetColumn, setTargetColumn] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [isDragging, setIsDragging] = useState(false)
    const fileInputRef = useRef(null)

    const parseColumns = (selectedFile) => {
        const reader = new FileReader()
        reader.onload = (e) => {
            const text = e.target.result
                .replace(/^\uFEFF/, '')   // strip BOM
                .trim()
            const firstLine = text.split(/\r?\n/)[0]
            // Handle quoted CSV headers
            const cols = firstLine.split(',').map(h =>
                h.trim().replace(/^["']|["']$/g, '')
            )
            setColumns(cols)
            // Default: last column (ML convention), but user can change
            setTargetColumn(cols[cols.length - 1])
        }
        reader.readAsText(selectedFile)
    }

    const handleFileChange = (selectedFile) => {
        if (!selectedFile) return
        if (!selectedFile.name.endsWith('.csv')) {
            setError('Only CSV files are supported')
            return
        }
        setFile(selectedFile)
        setError(null)
        parseColumns(selectedFile)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setIsDragging(false)
        handleFileChange(e.dataTransfer.files[0])
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!file || !targetColumn) return
        setLoading(true)
        setError(null)
        try {
            const formData = new FormData()
            formData.append('file', file)
            formData.append('target_column', targetColumn.trim())
            const res = await axios.post(`${API_BASE}/api/upload`, formData)
            onComplete(res.data)
        } catch (err) {
            if (!err.response) {
                setError('Backend is unreachable. Please ensure the backend server is running on port 8000.')
            } else {
                setError(err.response?.data?.detail || 'Upload failed. Check target column and CSV format.')
            }
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-slate-900">Upload Your Dataset</h1>
                <p className="mt-1 text-slate-500">
                    Upload a CSV file and select your <strong>target (output) column</strong>. We auto-detect the problem type.
                </p>
            </div>

            {existingData && (
                <div className="bg-green-50 border border-green-200 rounded-xl p-4 flex items-start gap-3">
                    <CheckCircle2 className="h-5 w-5 text-green-600 mt-0.5 shrink-0" />
                    <div>
                        <p className="font-semibold text-green-800">{existingData.message}</p>
                        <p className="text-sm text-green-700 mt-1">
                            {existingData.analysis.rows} rows · {existingData.analysis.columns} columns ·
                            Target: <strong>"{existingData.analysis.column_names?.slice(-1)[0]}"</strong>
                        </p>
                    </div>
                </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
                <div
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current.click()}
                    className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
            ${isDragging ? 'border-indigo-400 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}`}
                >
                    <input ref={fileInputRef} type="file" accept=".csv" className="hidden"
                        onChange={(e) => handleFileChange(e.target.files[0])} />
                    {file ? (
                        <div className="flex flex-col items-center gap-2">
                            <FileText className="h-10 w-10 text-indigo-500" />
                            <p className="font-semibold text-slate-800">{file.name}</p>
                            <p className="text-sm text-slate-500">{(file.size / 1024).toFixed(1)} KB · {columns.length} columns detected</p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-2">
                            <Upload className="h-10 w-10 text-slate-400" />
                            <p className="font-semibold text-slate-700">Drag & drop your CSV here</p>
                            <p className="text-sm text-slate-400">or click to browse</p>
                        </div>
                    )}
                </div>

                {columns.length > 0 && (
                    <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-3">
                        <label className="block text-sm font-medium text-slate-700">
                            🎯 Select Target Column <span className="text-slate-400 font-normal">(the variable to predict)</span>
                        </label>
                        <div className="relative">
                            <select
                                value={targetColumn}
                                onChange={(e) => setTargetColumn(e.target.value)}
                                className="w-full appearance-none bg-white border border-slate-300 rounded-lg px-4 py-2.5 text-slate-800 pr-10 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            >
                                {columns.map((col) => (
                                    <option key={col} value={col}>{col}</option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
                        </div>
                        <p className="text-xs text-slate-500">
                            Default is the last column (standard ML convention). Change if your target is elsewhere.
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                            {columns.map((col) => (
                                <button key={col} type="button" onClick={() => setTargetColumn(col)}
                                    className={`text-xs px-2.5 py-1 rounded-full border transition-colors
                    ${targetColumn === col
                                            ? 'bg-indigo-600 text-white border-indigo-600'
                                            : 'bg-white text-slate-600 border-slate-300 hover:border-indigo-400'}`}>
                                    {col}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {error && (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
                        <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                        <p className="text-red-700 text-sm">{error}</p>
                    </div>
                )}

                <button type="submit" disabled={!file || !targetColumn || loading}
                    className="w-full bg-indigo-600 text-white px-6 py-3 rounded-xl font-semibold text-sm hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2">
                    {loading ? (
                        <><span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" /> Analyzing...</>
                    ) : (
                        <><Upload className="h-4 w-4" /> Analyze Dataset</>
                    )}
                </button>
            </form>

            {existingData && (
                <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
                    <h2 className="font-semibold text-slate-800">Dataset Summary</h2>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        <StatCard label="Rows" value={existingData.analysis.rows} />
                        <StatCard label="Columns" value={existingData.analysis.columns} />
                        <StatCard label="Problem Type" value={existingData.analysis.problem_type} highlight />
                        <StatCard label="Missing Cells"
                            value={Object.values(existingData.analysis.missing_values || {}).reduce((a, b) => a + b, 0)} />
                    </div>
                    {existingData.analysis.class_distribution && (
                        <div>
                            <p className="text-sm font-medium text-slate-700 mb-2">Class Distribution</p>
                            <div className="flex flex-wrap gap-2">
                                {Object.entries(existingData.analysis.class_distribution).map(([cls, pct]) => (
                                    <span key={cls} className="bg-indigo-50 text-indigo-700 rounded-full px-3 py-1 text-xs font-medium">
                                        {cls}: {(pct * 100).toFixed(1)}%
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

function StatCard({ label, value, highlight }) {
    return (
        <div className={`rounded-lg p-4 text-center ${highlight ? 'bg-indigo-50' : 'bg-slate-50'}`}>
            <p className={`text-lg font-bold ${highlight ? 'text-indigo-700' : 'text-slate-800'}`}>{value}</p>
            <p className="text-xs text-slate-500 mt-1">{label}</p>
        </div>
    )
}
