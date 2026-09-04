
import React from "react";
import { 
  Flame, 
  AlertTriangle, 
  Activity, 
  MapPin, 
  Clock, 
  Satellite, 
  ShieldAlert, 
  Layers,
  ChevronRight
} from "lucide-react";

const formatFireType = (rawType) => {
  if (!rawType) return "Unknown Event";
  return rawType
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

const getRiskColor = (level) => {
  switch (level?.toUpperCase()) {
    case "CRITICAL":
      return {
        bg: "bg-red-500",
        text: "text-red-700",
        border: "border-red-300",
        lightBg: "bg-red-50",
        badge: "bg-red-600 text-white",
      };
    case "HIGH":
      return {
        bg: "bg-orange-500",
        text: "text-orange-700",
        border: "border-orange-300",
        lightBg: "bg-orange-50",
        badge: "bg-orange-600 text-white",
      };
    case "MODERATE":
      return {
        bg: "bg-amber-500",
        text: "text-amber-700",
        border: "border-amber-300",
        lightBg: "bg-amber-50",
        badge: "bg-amber-600 text-white",
      };
    case "LOW":
    default:
      return {
        bg: "bg-emerald-500",
        text: "text-emerald-700",
        border: "border-emerald-300",
        lightBg: "bg-emerald-50",
        badge: "bg-emerald-600 text-white",
      };
  }
};

const Sidebar = ({ setFirePannel, hotspot, prediction, isLoading, error }) => {
  const riskColors = getRiskColor(prediction?.risk_level);

  return (
    <div className="bg-white   rounded-[5px] p-4 md:p-5  text-black">
      {/* Header */}
      <div className="flex justify-between items-center pb-3 mb-3 border-b border-gray-100">
        <div className="flex items-center gap-2">
          
          <div>
            <h1 className="text-base md:text-lg font-bold text-gray-900 leading-tight">
              Thermal Analysis
            </h1>
            
          </div>
        </div>

        <button
          onClick={() => setFirePannel(false)}
          className="w-7 h-7 flex justify-center items-center bg-gray-100 hover:bg-gray-200 text-black rounded-full transition cursor-pointer"
          title="Close Sidebar"
        >
          ✕
        </button>
      </div>

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="space-y-3 animate-pulse py-2">
          <div className="h-16 bg-gray-200 rounded-xl"></div>
          <div className="h-12 bg-gray-200 rounded-xl"></div>
          <div className="grid grid-cols-2 gap-2">
            <div className="h-16 bg-gray-200 rounded-lg"></div>
            <div className="h-16 bg-gray-200 rounded-lg"></div>
          </div>
          <div className="h-24 bg-gray-200 rounded-xl"></div>
        </div>
      )}

      {/* Error Message */}
      {error && !isLoading && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700 flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Classification Unavailable</p>
            <p className="mt-0.5 text-gray-600">{error}</p>
          </div>
        </div>
      )}

      {/* Prediction Content */}
      {!isLoading && prediction && (
        <div className="space-y-3">
          {/* Main Fire Type & Risk Level Card */}
          <div className={`p-3.5 rounded-[5px] bg-slate-100 border-2 border-black  relative overflow-hidden`}>
            <div className="flex justify-between items-start">
              <div>
                <span className="text-[10px] uppercase tracking-wider font-semibold text-gray-500">
                  Fire Type
                </span>
                <h2 className="text-base md:text-lg font-extrabold text-gray-900 leading-snug">
                  {formatFireType(prediction.predicted_fire_type)}
                </h2>
              </div>
              <span className={`px-2.5 py-0.5 text-xs  rounded-[5px] ${riskColors.badge} uppercase tracking-wider shadow-sm`}>
                {prediction.risk_level} RISK
              </span>
            </div>

            {/* Confidence & Risk Scores */}
            <div className="mt-3 pt-2.5 border-t border-gray-200/60 grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500 block text-[11px]">AI Confidence</span>
                <span className="font-bold text-gray-900 text-sm">
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
                <div className="w-full bg-gray-200 h-1.5 rounded-[2px] mt-1 overflow-hidden">
                  <div
                    className="bg-gray-600 h-full rounded-[2px]"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <span className="text-gray-500 block text-[11px]">Risk Score</span>
                <span className="font-bold text-gray-900 text-sm">
                  {prediction.risk_score} <span className="text-[10px] text-gray-400">/ 100</span>
                </span>
                <div className="w-full bg-gray-200 h-1.5 rounded-[2px] mt-1 overflow-hidden">
                  <div
                    className={`${riskColors.bg} h-full rounded-[2px]`}
                    style={{ width: `${prediction.risk_score}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Satellite Telemetry Grid */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-100 border-black border-2  p-2.5 rounded-[5px]">
              <div className="flex items-center gap-1.5 text-gray-500 mb-0.5">
                <Activity className="w-3.5 h-3.5 text-black" />
                <span>Fire Power (FRP)</span>
              </div>
              <p className="font-bold text-gray-900 text-sm">
                {prediction.summary?.avg_frp_mw || hotspot?.frp || 0} <span className="text-[10px] font-normal text-gray-500">MW</span>
              </p>
            </div>

            <div className="bg-slate-100 border-2 border-black p-2.5 rounded-[5px]">
              <div className="flex items-center gap-1.5 text-gray-500 mb-0.5">
                <Flame className="w-3.5 h-3.5 text-black" />
                <span>Brightness Temp</span>
              </div>
              <p className="font-bold text-gray-900 text-sm">
                {prediction.summary?.bright_ti4_k || hotspot?.bright_ti4 || 0} <span className="text-[10px] font-normal text-gray-500">K</span>
              </p>
            </div>

            <div className="bg-slate-100 border-2 border-black  p-2.5 rounded-[5px]">
              <div className="flex items-center gap-1.5 text-gray-500 mb-0.5">
                <MapPin className="w-3.5 h-3.5 text-black" />
                <span>Coordinates</span>
              </div>
              <p className="font-semibold text-gray-800 text-[11px]">
                {prediction.latitude.toFixed(3)}°N, {prediction.longitude.toFixed(3)}°E
              </p>
            </div>

            <div className="bg-slate-100 border-2 border-black  p-2.5 rounded-[5px]">
              <div className="flex items-center gap-1.5 text-gray-500 mb-0.5">
                <Satellite className="w-3.5 h-3.5 text-black" />
                <span>Sensor</span>
              </div>
              <p className="font-semibold text-gray-800 text-[11px] truncate">
                {hotspot?.satellite || "VIIRS NOAA-20"} ({hotspot?.daynight === "N" ? "Night" : "Day"})
              </p>
            </div>
          </div>

          {/* Class Probability Distribution */}
          {prediction.class_probabilities && (
            <div className="bg-gray-50 border-2 border-black p-3 rounded-[5px]">
              <h3 className="text-[11px] font-bold text-gray-600 uppercase tracking-wider mb-2 flex items-center gap-1">
                <Layers className="w-3 h-3 text-gray-500" />
                Probabilities
              </h3>
              <div className="space-y-1.5">
                {Object.entries(prediction.class_probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([cls, prob]) => (
                    <div key={cls} className="text-[11px]">
                      <div className="flex justify-between text-gray-700 mb-0.5">
                        <span className="truncate pr-2">{formatFireType(cls)}</span>
                        <span className="font-semibold">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 h-1 rounded-[2px] overflow-hidden">
                        <div
                          className="bg-gray-700 h-full rounded-[2px]"
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          
        </div>
      )}
    </div>
  );
};

export default Sidebar;