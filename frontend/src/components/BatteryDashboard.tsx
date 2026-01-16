import { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    AreaChart, Area
} from 'recharts';
import { Battery, Zap, AlertCircle, Activity, TrendingUp, PlayCircle, Pause } from 'lucide-react';

interface SensorData {
    time: string;
    value: number; // Generic key for user's snippet compatibility map
    RealPower: number;
    PowerDifference: number;
    GateOnTime: number;
    Length: number;
    Speed: number;
    prediction: string;
}

export function BatteryDashboard() {
    const [isMonitoring, setIsMonitoring] = useState(false);
    const [systemStatus, setSystemStatus] = useState<'WAITING' | 'MONITORING' | 'ANALYZING'>('WAITING');

    // KPI Stats (Maintained from my previous logic)
    const [stats, setStats] = useState({
        totalCount: 1250,
        ngCount: 12,
        efficiency: 98.2,
        recentYield: 99.5
    });

    // Real-time Data Buffer
    const [dataBuffer, setDataBuffer] = useState<SensorData[]>([]);

    // Simulation Config (My previous logic)
    const SIM_CONFIG = {
        OK: {
            RealPower: { mean: 150, std: 2 },
            Length: { mean: 45, std: 1 },
            GateOnTime: { mean: 85, std: 2 },
            Speed: { mean: 250, std: 2 }
        },
        NG: {
            RealPower: { mean: 130, std: 10 },
            Length: { mean: 40, std: 3 },
            GateOnTime: { mean: 95, std: 5 },
            Speed: { mean: 260, std: 5 }
        }
    };

    const generateDataPoint = () => {
        // Simple random choice for simulation
        const type = Math.random() > 0.05 ? 'OK' : 'NG';
        const config = SIM_CONFIG[type];

        const rand = (mean: number, std: number) => {
            let u = 0, v = 0;
            while (u === 0) u = Math.random();
            while (v === 0) v = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            return mean + z * std;
        };

        const RealPower = Math.round(rand(config.RealPower.mean, config.RealPower.std));
        const Length = parseFloat(rand(config.Length.mean, config.Length.std).toFixed(1));
        const GateOnTime = Math.round(rand(config.GateOnTime.mean, config.GateOnTime.std));
        const Speed = Math.round(rand(config.Speed.mean, config.Speed.std));
        const PowerDifference = Math.abs(Math.round(rand(8, 3))); // Target < 15

        return { type, Speed, Length, RealPower, GateOnTime, PowerDifference };
    };

    const handleStartMonitoring = () => {
        setIsMonitoring(true);
        setSystemStatus('MONITORING');
        setTimeout(() => {
            setSystemStatus('ANALYZING');
        }, 2000);
    };

    const handleStopMonitoring = () => {
        setIsMonitoring(false);
        setSystemStatus('WAITING');
    };

    useEffect(() => {
        let interval: NodeJS.Timeout;

        if (isMonitoring) {
            // My data generation loop
            interval = setInterval(() => {
                const newData = generateDataPoint();
                const now = new Date();
                const timeStr = `${now.getSeconds()}s`;

                const newEntry: SensorData = {
                    time: timeStr,
                    value: 0, // Placeholder
                    RealPower: newData.RealPower,
                    PowerDifference: newData.PowerDifference,
                    GateOnTime: newData.GateOnTime,
                    Length: newData.Length,
                    Speed: newData.Speed,
                    prediction: newData.type
                };

                // Update Stats
                setStats(prev => ({
                    totalCount: prev.totalCount + 1,
                    ngCount: newData.type === 'NG' ? prev.ngCount + 1 : prev.ngCount,
                    efficiency: prev.efficiency, // Keep static or slowly drift
                    recentYield: parseFloat(((1 - (newData.type === 'NG' ? prev.ngCount + 1 : prev.ngCount) / (prev.totalCount + 1)) * 100).toFixed(1))
                }));

                setDataBuffer(prev => {
                    const newBuffer = [...prev, newEntry];
                    if (newBuffer.length > 10) newBuffer.shift(); // Keep only last 10 points like user snippet
                    return newBuffer;
                });
            }, 2000);
        }

        return () => clearInterval(interval);
    }, [isMonitoring]);

    return (
        <div className="p-8 bg-gradient-to-br from-gray-50 to-blue-50 min-h-screen">
            {/* Header */}
            <div className="mb-8 flex items-center justify-between">
                <div>
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                            <Battery className="w-7 h-7 text-white" />
                        </div>
                        <div>
                            <h2 className="text-3xl font-bold text-gray-900">배터리 모니터링</h2>
                            <p className="text-gray-600 mt-1">배터리 불량 분석</p>
                        </div>
                    </div>
                </div>
                <div className="flex gap-3">
                    {!isMonitoring ? (
                        <button
                            onClick={handleStartMonitoring}
                            className="px-6 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl flex items-center gap-2"
                        >
                            <PlayCircle className="w-5 h-5" />
                            START MONITORING
                        </button>
                    ) : (
                        <button
                            onClick={handleStopMonitoring}
                            className="px-6 py-3 bg-red-600 text-white rounded-xl font-semibold hover:bg-red-700 transition-all shadow-lg hover:shadow-xl flex items-center gap-2"
                        >
                            <Pause className="w-5 h-5" />
                            STOP MONITORING
                        </button>
                    )}
                    <button className="px-4 py-3 bg-white border-2 border-gray-200 rounded-xl font-semibold hover:bg-gray-50 transition-all">
                        ⚙️
                    </button>
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <Battery className="w-5 h-5 text-green-600" />
                            <p className="text-sm font-medium text-gray-600">Battery Health</p>
                        </div>
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    </div>
                    <p className="text-4xl font-bold text-gray-900 mb-2">{stats.efficiency}%</p>
                    <p className="text-xs text-green-600 font-medium">▲ 0.5% from last week</p>
                </div>

                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <Activity className="w-5 h-5 text-blue-600" />
                            <p className="text-sm font-medium text-gray-600">Total Inspected</p>
                        </div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                    </div>
                    <p className="text-4xl font-bold text-gray-900 mb-2">{stats.totalCount}</p>
                    <p className="text-xs text-blue-600 font-medium">+127 today</p>
                </div>

                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <AlertCircle className="w-5 h-5 text-orange-600" />
                            <p className="text-sm font-medium text-gray-600">Defect Detected</p>
                        </div>
                        <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse" />
                    </div>
                    <p className="text-4xl font-bold text-gray-900 mb-2">{stats.ngCount}</p>
                    <p className="text-xs text-orange-600 font-medium">-3 from yesterday</p>
                </div>

                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <TrendingUp className="w-5 h-5 text-purple-600" />
                            <p className="text-sm font-medium text-gray-600">Real-time Yield</p>
                        </div>
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                    </div>
                    <p className="text-4xl font-bold text-gray-900 mb-2">{stats.recentYield}%</p>
                    <p className="text-xs text-purple-600 font-medium">Excellent performance</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                {/* System Status */}
                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-900 mb-6">System Status</h3>
                    <div className="flex flex-col items-center justify-center py-12">
                        <div className={`w-24 h-24 rounded-2xl flex items-center justify-center mb-4 ${systemStatus === 'WAITING' ? 'bg-gray-100' :
                            systemStatus === 'MONITORING' ? 'bg-blue-100 animate-pulse' :
                                'bg-green-100 animate-pulse'
                            }`}>
                            <Activity className={`w-12 h-12 ${systemStatus === 'WAITING' ? 'text-gray-400' :
                                systemStatus === 'MONITORING' ? 'text-blue-600' :
                                    'text-green-600'
                                }`} />
                        </div>
                        <p className={`text-2xl font-bold mb-2 ${systemStatus === 'WAITING' ? 'text-gray-400' :
                            systemStatus === 'MONITORING' ? 'text-blue-600' :
                                'text-green-600'
                            }`}>
                            {systemStatus}
                        </p>
                        <p className="text-sm text-gray-500 text-center">
                            {systemStatus === 'WAITING' && 'Ready to start monitoring'}
                            {systemStatus === 'MONITORING' && 'Collecting data...'}
                            {systemStatus === 'ANALYZING' && 'AI analysis in progress'}
                        </p>
                    </div>
                </div>

                {/* Analysis Insight */}
                <div className="lg:col-span-2 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl shadow-sm p-6 border-2 border-blue-200">
                    <div className="flex items-center gap-2 mb-4">
                        <Zap className="w-5 h-5 text-blue-600" />
                        <h3 className="text-lg font-bold text-gray-900">Real-time Failure Importance Interpretation</h3>
                    </div>
                    <div className="mb-3">
                        <p className="text-sm text-gray-600 font-medium mb-2">Model: Random Forest v2.1</p>
                    </div>
                    <div className="bg-white rounded-xl p-5 shadow-sm border border-blue-100">
                        <div className="flex items-start gap-3">
                            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                                <Zap className="w-5 h-5 text-blue-600" />
                            </div>
                            <div className="flex-1">
                                <h4 className="font-bold text-gray-900 mb-2">Analysis Insight</h4>
                                <p className="text-sm text-gray-700 leading-relaxed">
                                    분석 결과, <span className="font-bold text-blue-600">RealPower</span>와{' '}
                                    <span className="font-bold text-blue-600">PowerDifference</span>의 변동이 클수록 용접 불량 위험이 높아지는 것으로 확인되었습니다.
                                </p>
                                <div className="mt-4 pt-4 border-t border-gray-200 grid grid-cols-2 gap-4">
                                    <div>
                                        <p className="text-xs text-gray-500 mb-1">Key Factor 1</p>
                                        <p className="text-sm font-bold text-gray-900">Real Power (87% impact)</p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-gray-500 mb-1">Key Factor 2</p>
                                        <p className="text-sm font-bold text-gray-900">Power Diff (76% impact)</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Charts - Using dataBuffer for dynamic data */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Real Power */}
                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-blue-500 rounded-full" />
                            <h3 className="text-lg font-bold text-gray-900">Real Power</h3>
                        </div>
                        <span className="px-3 py-1 bg-gray-900 text-white text-xs font-bold rounded-full">
                            target: 150
                        </span>
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <AreaChart data={dataBuffer}>
                            <defs>
                                <linearGradient id="colorRealPower" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="time" stroke="#6b7280" style={{ fontSize: '12px' }} />
                            <YAxis domain={[120, 180]} stroke="#6b7280" style={{ fontSize: '12px' }} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'white',
                                    border: '1px solid #e5e7eb',
                                    borderRadius: '8px',
                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                }}
                            />
                            <Area type="monotone" dataKey="RealPower" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorRealPower)" isAnimationActive={false} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Power Difference */}
                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-purple-500 rounded-full" />
                            <h3 className="text-lg font-bold text-gray-900">Power Difference</h3>
                        </div>
                        <span className="px-3 py-1 bg-gray-900 text-white text-xs font-bold rounded-full">
                            target: &lt;15
                        </span>
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={dataBuffer}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="time" stroke="#6b7280" style={{ fontSize: '12px' }} />
                            <YAxis stroke="#6b7280" style={{ fontSize: '12px' }} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'white',
                                    border: '1px solid #e5e7eb',
                                    borderRadius: '8px',
                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                }}
                            />
                            <Line type="monotone" dataKey="PowerDifference" stroke="#a855f7" strokeWidth={3} dot={{ fill: '#a855f7', r: 4 }} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Gate On Time */}
                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full" />
                            <h3 className="text-lg font-bold text-gray-900">Gate On Time</h3>
                        </div>
                        <span className="px-3 py-1 bg-gray-900 text-white text-xs font-bold rounded-full">
                            target: 85±5
                        </span>
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <AreaChart data={dataBuffer}>
                            <defs>
                                <linearGradient id="colorGateOn" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="time" stroke="#6b7280" style={{ fontSize: '12px' }} />
                            <YAxis domain={[75, 100]} stroke="#6b7280" style={{ fontSize: '12px' }} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'white',
                                    border: '1px solid #e5e7eb',
                                    borderRadius: '8px',
                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                }}
                            />
                            <Area type="monotone" dataKey="GateOnTime" stroke="#22c55e" strokeWidth={3} fillOpacity={1} fill="url(#colorGateOn)" isAnimationActive={false} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Weld Length */}
                <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-200">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-orange-500 rounded-full" />
                            <h3 className="text-lg font-bold text-gray-900">Weld Length</h3>
                        </div>
                        <span className="px-3 py-1 bg-gray-900 text-white text-xs font-bold rounded-full">
                            target: ≥43
                        </span>
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={dataBuffer}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="time" stroke="#6b7280" style={{ fontSize: '12px' }} />
                            <YAxis domain={[35, 55]} stroke="#6b7280" style={{ fontSize: '12px' }} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'white',
                                    border: '1px solid #e5e7eb',
                                    borderRadius: '8px',
                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                }}
                            />
                            <Line type="monotone" dataKey="Length" stroke="#f97316" strokeWidth={3} dot={{ fill: '#f97316', r: 4 }} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
