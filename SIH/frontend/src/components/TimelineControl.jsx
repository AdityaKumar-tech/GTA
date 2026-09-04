import React from "react";
import { 
  Clock, 
  Flame, 
  Trees, 
  Wheat, 
  Factory, 
  Pickaxe, 
  Zap, 
  Warehouse, 
  Sprout, 
  Building2, 
  Truck, 
  Mountain, 
  HelpCircle 
} from "lucide-react";

const PRESETS = [
  { id: "live", label: "⭕️ Live (24h)", days: "live" },
  { id: "7d", label: "7 Days", days: "7d" },
  { id: "30d", label: "30 Days", days: "30d" },
];

export const FIRE_TYPE_CONFIG = [
  { id: "ALL", label: "All Types", icon: Flame, color: "text-rose-500", bg: "bg-rose-500", hex: "#f43f5e" },
  { id: "INDUSTRIAL_FIRE", label: "Industrial", icon: Factory, color: "text-blue-600", bg: "bg-blue-600", hex: "#2563eb" },
  { id: "FOREST_WILDFIRE", label: "Forest Wildfire", icon: Trees, color: "text-emerald-600", bg: "bg-emerald-600", hex: "#059669" },
  { id: "CROP_RESIDUE_FIRE", label: "Crop Residue", icon: Wheat, color: "text-amber-500", bg: "bg-amber-500", hex: "#d97706" },
  { id: "MINING_RELATED_FIRE", label: "Mining", icon: Pickaxe, color: "text-red-600", bg: "bg-red-600", hex: "#dc2626" },
  { id: "INFRASTRUCTURE_RELATED_FIRE", label: "Infrastructure", icon: Zap, color: "text-cyan-600", bg: "bg-cyan-600", hex: "#0891b2" },
  { id: "FACILITY_RELATED_FIRE", label: "Facility", icon: Warehouse, color: "text-indigo-600", bg: "bg-indigo-600", hex: "#4f46e5" },
  { id: "GRASSLAND_FIRE", label: "Grassland", icon: Sprout, color: "text-green-500", bg: "bg-green-500", hex: "#16a34a" },
  { id: "URBAN_FIRE", label: "Urban", icon: Building2, color: "text-purple-600", bg: "bg-purple-600", hex: "#7c3aed" },
  { id: "TRANSPORTATION_RELATED_FIRE", label: "Transportation", icon: Truck, color: "text-orange-500", bg: "bg-orange-500", hex: "#ea580c" },
  { id: "OPEN_LAND_FIRE", label: "Open Land", icon: Mountain, color: "text-stone-500", bg: "bg-stone-500", hex: "#78716c" },
  { id: "UNKNOWN_THERMAL_EVENT", label: "Unknown", icon: HelpCircle, color: "text-gray-500", bg: "bg-gray-500", hex: "#6b7280" },
];

const TimelineControl = ({
  activePreset,
  setActivePreset,
  activeFilter,
  setActiveFilter,
  totalMatches,
  displayedCount,
  isLoading
}) => {
  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 w-11/12 max-w-5xl bg-white  rounded-[5px]  border border-gray-200/90 p-2.5 md:p-3 text-black transition-all">
      {/* Top Header & Presets */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-gray-100">
        <div className="flex items-center gap-2">
          
          <span className="text-xs md:text-sm font-bold text-gray-900 block leading-tight">
            Timeline:
          </span>
          <div className="flex items-center bg-gray-100 p-0.5 rounded-lg gap-1">
            {PRESETS.map((preset) => {
              const isActive = activePreset === preset.id;
              return (
                <button
                  key={preset.id}
                  onClick={() => setActivePreset(preset.id)}
                  className={`px-2.5 py-1 text-xs font-semibold rounded-md transition cursor-pointer ${
                    isActive
                      ? "bg-white text-gray-900 shadow-sm border border-gray-200"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-200/60"
                  }`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Counter Badge */}
        <div className="flex items-center gap-2 text-xs">
          {isLoading ? (
            <span className="text-gray-500 animate-pulse text-[11px]">Loading satellite data...</span>
          ) : (
            <div className="bg-gray-100 px-2.5 py-1 rounded-md border border-gray-200 font-medium text-gray-700 flex items-center gap-1 text-[11px]">
              <span>Showing:</span>
              <span className="font-bold text-gray-900">{displayedCount}</span>
              {totalMatches > displayedCount && (
                <span className="text-gray-500">
                  (of <strong className="text-gray-800">{totalMatches.toLocaleString()}</strong>)
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Bottom 11 Fire Type Filter Strip */}
      <div className="flex items-center gap-2 pt-2 overflow-x-auto no-scrollbar scroll-smooth">
        <span className="text-xs md:text-sm font-bold w-17 shrink-0">
          Filter:
        </span>
        <div className="flex items-center gap-1.5 shrink-0">
          {FIRE_TYPE_CONFIG.map((filter) => {
            const Icon = filter.icon;
            const isActive = activeFilter === filter.id;
            return (
              <button
                key={filter.id}
                onClick={() => setActiveFilter(filter.id)}
                className={`px-2.5 py-1 text-[11px] font-medium rounded-md flex items-center gap-1.5 transition cursor-pointer border shrink-0 ${
                  isActive
                    ? "bg-gray-900 text-white border-gray-900 shadow-sm"
                    : "bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100"
                }`}
              >
                <Icon className={`w-3 h-3 ${isActive ? "text-white" : filter.color}`} />
                <span>{filter.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default TimelineControl;
