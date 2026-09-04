import { useEffect, useRef, useState, useCallback } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { Crosshair, MapPin, Layers } from "lucide-react";

const API_BASE_URL = "http://localhost:8000/api/v1";

const Map = ({
  onSelectHotspot,
  searchLocation,
  activePreset = "live",
  activeFilter = "ALL",
  setTotalMatches,
  setDisplayedCount,
  setIsLoadingHotspots,
  isLoadingHotspots
}) => {
  const mapContainer = useRef(null);
  const mapInstance = useRef(null);
  const isMapLoaded = useRef(false);
  const searchMarkerRef = useRef(null);

  // Real-time cursor coordinates and zoom level (NASA FIRMS style)
  const [cursorCoords, setCursorCoords] = useState({
    lat: "21.5937",
    lng: "78.9629",
    zoom: "4.8"
  });
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const loadData = useCallback(async () => {
    if (!mapInstance.current || !isMapLoaded.current) return;

    try {
      if (setIsLoadingHotspots) setIsLoadingHotspots(true);

      let url = "";
      if (activePreset === "live") {
        url = `${API_BASE_URL}/firms/live?source=ALL&day_range=1&fire_type=${activeFilter}`;
      } else {
        url = `${API_BASE_URL}/timeline?preset=${activePreset}&fire_type=${activeFilter}&limit=50000`;
      }

      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to load map data");
      const geojsonData = await res.json();

      const count = geojsonData.count || (geojsonData.features ? geojsonData.features.length : 0);
      const total = geojsonData.total_matches || count;

      if (setDisplayedCount) setDisplayedCount(count);
      if (setTotalMatches) setTotalMatches(total);

      const source = mapInstance.current.getSource("live-hotspots");
      if (source) {
        source.setData(geojsonData);
      }
    } catch (err) {
      console.error("Error loading map data:", err);
    } finally {
      if (setIsLoadingHotspots) setIsLoadingHotspots(false);
    }
  }, [activePreset, activeFilter, setDisplayedCount, setTotalMatches, setIsLoadingHotspots]);

  useEffect(() => {
    const mapTilerKey = import.meta.env.VITE_MAPTILER_KEY;
    const mapStyle = mapTilerKey
      ? `https://api.maptiler.com/maps/hybrid/style.json?key=${mapTilerKey}`
      : "https://demotiles.maplibre.org/style.json";

    const map = new maplibregl.Map({
      container: mapContainer.current,
      center: [78.9629, 21.5937],
      zoom: 4.8,
      style: mapStyle,
    });

    mapInstance.current = map;
    map.addControl(new maplibregl.NavigationControl(), "top-left");

    // Track real-time mouse position & zoom level across the map
    map.on("mousemove", (e) => {
      const { lng, lat } = e.lngLat;
      const currentZoom = map.getZoom();
      setCursorCoords({
        lat: lat.toFixed(4),
        lng: lng.toFixed(4),
        zoom: currentZoom.toFixed(1)
      });
    });

    map.on("zoom", () => {
      setCursorCoords((prev) => ({
        ...prev,
        zoom: map.getZoom().toFixed(1)
      }));
    });

    map.on("load", () => {
      isMapLoaded.current = true;

      // Initialize empty GeoJSON source
      map.addSource("live-hotspots", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });

      // 1. Hotspot outer glow layer
      map.addLayer({
        id: "hotspots-glow",
        type: "circle",
        source: "live-hotspots",
        paint: {
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["get", "frp"],
            5, 8,
            20, 15,
            60, 24,
          ],
          "circle-color": [
            "match",
            ["get", "fire_type"],
            "INDUSTRIAL_FIRE", "#3b82f6",          // Blue
            "FOREST_WILDFIRE", "#10b981",          // Emerald Green
            "CROP_RESIDUE_FIRE", "#f59e0b",        // Amber
            "MINING_RELATED_FIRE", "#ef4444",      // Red
            "INFRASTRUCTURE_RELATED_FIRE", "#06b6d4", // Cyan
            "FACILITY_RELATED_FIRE", "#818cf8",    // Indigo
            "GRASSLAND_FIRE", "#4ade80",           // Mint Green
            "URBAN_FIRE", "#a855f7",               // Purple
            "TRANSPORTATION_RELATED_FIRE", "#fb923c", // Orange
            "OPEN_LAND_FIRE", "#a8a29e",           // Stone
            "UNKNOWN_THERMAL_EVENT", "#9ca3af",    // Gray
            /* fallback */
            "#f97316"
          ],
          "circle-opacity": 0.55,
          "circle-blur": 0.6,
        },
      });

      // 2. Hotspot inner core layer
      map.addLayer({
        id: "hotspots-core",
        type: "circle",
        source: "live-hotspots",
        paint: {
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["get", "frp"],
            5, 4.5,
            20, 7.5,
            60, 12,
          ],
          "circle-color": [
            "match",
            ["get", "fire_type"],
            "INDUSTRIAL_FIRE", "#2563eb",          // Royal Blue
            "FOREST_WILDFIRE", "#059669",          // Forest Green
            "CROP_RESIDUE_FIRE", "#d97706",        // Amber
            "MINING_RELATED_FIRE", "#dc2626",      // Crimson Red
            "INFRASTRUCTURE_RELATED_FIRE", "#0891b2", // Cyan
            "FACILITY_RELATED_FIRE", "#4f46e5",    // Indigo
            "GRASSLAND_FIRE", "#16a34a",           // Green
            "URBAN_FIRE", "#7c3aed",               // Purple
            "TRANSPORTATION_RELATED_FIRE", "#ea580c", // Orange
            "OPEN_LAND_FIRE", "#78716c",           // Stone
            "UNKNOWN_THERMAL_EVENT", "#4b5563",    // Gray
            /* fallback */
            "#ea580c"
          ],
          "circle-stroke-width": 1.5,
          "circle-stroke-color": "#ffffff",
          "circle-opacity": 0.95,
        },
      });

      // Cursor and Click interactions
      map.on("mouseenter", "hotspots-core", (e) => {
        map.getCanvas().style.cursor = "pointer";
        if (e.features && e.features.length > 0) {
          setHoveredPoint(e.features[0].properties);
        }
      });

      map.on("mouseleave", "hotspots-core", () => {
        map.getCanvas().style.cursor = "";
        setHoveredPoint(null);
      });

      map.on("click", "hotspots-core", (e) => {
        if (e.features && e.features.length > 0) {
          const props = e.features[0].properties;
          if (onSelectHotspot) {
            onSelectHotspot(props);
          }
        }
      });

      // Fetch initial data
      loadData();
    });

    return () => {
      map.remove();
    };
  }, []);

  // Reload when timeline or filter changes
  useEffect(() => {
    if (isMapLoaded.current) {
      loadData();
    }
  }, [loadData]);

  // Handle FlyTo and Marker on Location Search
  useEffect(() => {
    if (!mapInstance.current || !searchLocation) return;
    const { lng, lat, zoom = 11, name } = searchLocation;

    mapInstance.current.flyTo({
      center: [lng, lat],
      zoom: zoom,
      duration: 2200,
      essential: true,
    });

    // Remove existing search pin marker
    if (searchMarkerRef.current) {
      searchMarkerRef.current.remove();
    }

    const popup = new maplibregl.Popup({ offset: [0, -42], closeButton: true }).setHTML(
      `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:4px 6px;min-width:140px;line-height:1.4;">
        <div style="font-weight:700;color:#202124;font-size:13px;margin-bottom:2px;">📍 ${name.split(',')[0]}</div>
        <div style="color:#5f6368;font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${name}</div>
        <div style="color:#1a73e8;font-size:10.5px;font-weight:600;margin-top:4px;font-family:monospace;">${lat.toFixed(4)}°N, ${lng.toFixed(4)}°E</div>
      </div>`
    );

    const el = document.createElement('div');
    el.className = 'google-maps-pin-marker';
    el.innerHTML = `
      <div style="position:relative;width:34px;height:44px;display:flex;flex-direction:column;align-items:center;cursor:pointer;filter:drop-shadow(0 4px 6px rgba(0,0,0,0.35));transition:transform 0.2s ease;">
        <svg viewBox="0 0 24 24" width="34" height="44" style="overflow:visible;display:block;">
          <!-- Red Google Maps Pin Body -->
          <path fill="#EA4335" stroke="#C5221F" stroke-width="0.8" d="M12 0C7.58 0 4 3.58 4 8c0 5.4 8 15 8 15s8-9.6 8-15c0-4.42-3.58-8-8-8z"/>
          <!-- Inner White Center Disc -->
          <circle cx="12" cy="8" r="3.2" fill="#FFFFFF"/>
          <!-- Inner Core Dot -->
          <circle cx="12" cy="8" r="1.4" fill="#EA4335"/>
        </svg>
        <!-- Ground Shadow -->
        <div style="position:absolute;bottom:-3px;width:14px;height:5px;background:radial-gradient(ellipse at center, rgba(0,0,0,0.45) 0%, rgba(0,0,0,0) 75%);border-radius:50%;z-index:-1;"></div>
      </div>
    `;

    const marker = new maplibregl.Marker({ element: el, anchor: "bottom" })
      .setLngLat([lng, lat])
      .setPopup(popup)
      .addTo(mapInstance.current);

    popup.addTo(mapInstance.current);
    searchMarkerRef.current = marker;
  }, [searchLocation]);

  return (
    <div className="relative w-full h-screen">
      <div
        ref={mapContainer}
        className="w-full h-full [&_.maplibregl-ctrl-top-left]:mt-[75px]"
      />

      {/* Floating Mode Status Badge (Top Left) */}
      <div className="absolute top-20 left-16 z-30 bg-white/90 backdrop-blur-md px-3.5 py-2 rounded-[5px] shadow-lg border border-gray-200 text-xs md:text-sm font-medium flex items-center gap-2">
        <span className="relative flex h-3 w-3">
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${activePreset === "live" ? "bg-red-400" : "bg-blue-700"} opacity-75`}></span>
          <span className={`relative inline-flex rounded-full h-3 w-3 ${activePreset === "live" ? "bg-red-700" : "bg-blue-700"}`}></span>
        </span>
        <div className="flex items-center gap-1 text-gray-800">
          <span className="font-medium text-gray-900">
            {activePreset === "live" ? "Live Satellite Hotspots" : `Historical Archive (${activePreset.toUpperCase()})`}
          </span>
          {activeFilter !== "ALL" && (
            <span className="bg-gray-100 text-gray-600 px-2 py-0.5 rounded text-[11px] font-semibold">
              {activeFilter.replace(/_/g, " ")}
            </span>
          )}
        </div>
      </div>

      {/* NASA FIRMS Style Live Coordinates HUD (Top Middle under Navbar) */}
      <div className="absolute top-[72px] left-1/2 -translate-x-1/2 z-30 bg-gray-900 text-white px-4 py-1.5 rounded-[5px] text-[11px] md:text-xs font-mono flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-emerald-400">
          <Crosshair className="w-3.5 h-3.5" />
          <span>LAT: <strong className="text-white font-mono">{cursorCoords.lat}°N</strong></span>
        </div>
        <span className="text-gray-500">|</span>
        <div className="flex items-center gap-1.5 text-emerald-400">
          <span>LON: <strong className="text-white font-mono">{cursorCoords.lng}°E</strong></span>
        </div>
        <span className="text-gray-500">|</span>
        <div className="text-amber-400 font-semibold flex items-center gap-1">
          <span>ZOOM:</span>
          <span className="text-white font-mono">{cursorCoords.zoom}z</span>
        </div>
      </div>

      {/* Map Fire Type Legend (Bottom Right - 11 Classes Key) */}
      <div className="absolute bottom-40 md:bottom-40 left-4 z-30 bg-white px-3.5 py-2.5 rounded-[5px] text-[11px] text-gray-700 hidden sm:flex flex-col gap-1.5 max-h-[320px] overflow-y-auto no-scrollbar">
        <span className="font-bold text-gray-900 text-[10px] uppercase tracking-wider border-b border-gray-100 pb-1">
          Fire Types
        </span>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-600 shrink-0"></span>
            <span className="truncate">Industrial</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 shrink-0"></span>
            <span className="truncate">Forest Wildfire</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shrink-0"></span>
            <span className="truncate">Crop Residue</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-600 shrink-0"></span>
            <span className="truncate">Mining Related</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-600 shrink-0"></span>
            <span className="truncate">Infrastructure</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-600 shrink-0"></span>
            <span className="truncate">Facility</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-green-500 shrink-0"></span>
            <span className="truncate">Grassland</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-purple-600 shrink-0"></span>
            <span className="truncate">Urban Fire</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-orange-500 shrink-0"></span>
            <span className="truncate">Transportation</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-stone-500 shrink-0"></span>
            <span className="truncate">Open Land</span>
          </div>
          <div className="flex items-center gap-1.5 col-span-2">
            <span className="w-2.5 h-2.5 rounded-full bg-gray-500 shrink-0"></span>
            <span className="truncate">Unknown Event</span>
          </div>
        </div>
      </div>

      {/* Hover Tooltip Card (When hovering over any hotspot) */}
      {hoveredPoint && (
        <div className="absolute top-20 right-4 z-30 bg-gray-900/90 backdrop-blur-md text-white px-3 py-2 rounded-[5px] shadow-2xl border border-gray-700 text-xs flex items-center gap-2 animate-fade-in pointer-events-none">
          <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping"></span>
          <div>
            <div className="font-bold text-blue-400">
              {hoveredPoint.fire_type ? hoveredPoint.fire_type.replace(/_/g, " ") : "Active Thermal Hotspot"}
            </div>
            <div className="text-[10px] text-gray-300">
              FRP: {hoveredPoint.frp || hoveredPoint.avg_frp || 0} MW • Click to view AI analysis
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Map;