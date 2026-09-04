import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { Search, MapPin, X, Loader2, Navigation } from "lucide-react";
import logo from "../assets/logo.png";

const Navbar = ({ onSelectLocation }) => {
  const [isSearchExpanded, setIsSearchExpanded] = useState(false);
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const searchContainerRef = useRef(null);
  const debounceTimerRef = useRef(null);

  const mapTilerKey = import.meta.env.VITE_MAPTILER_KEY;

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Debounced search query handler
  useEffect(() => {
    if (!query.trim() || query.trim().length < 2) {
      setSuggestions([]);
      setIsOpen(false);
      setIsLoading(false);
      return;
    }

    // Check if user entered direct coordinates like "21.14, 79.08"
    const coordMatch = query.match(/^([-+]?\d+(\.\d+)?)[,\s]+([-+]?\d+(\.\d+)?)$/);
    if (coordMatch) {
      const lat = parseFloat(coordMatch[1]);
      const lng = parseFloat(coordMatch[3]);
      if (lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
        setSuggestions([
          {
            id: "coord",
            name: `Coordinates: ${lat.toFixed(4)}°N, ${lng.toFixed(4)}°E`,
            lat: lat,
            lng: lng,
            zoom: 12,
            isCoord: true
          }
        ]);
        setIsOpen(true);
        return;
      }
    }

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    setIsLoading(true);
    debounceTimerRef.current = setTimeout(async () => {
      try {
        let results = [];

        // 1. Try MapTiler Geocoding API if key is available
        if (mapTilerKey) {
          const res = await fetch(
            `https://api.maptiler.com/geocoding/${encodeURIComponent(query)}.json?key=${mapTilerKey}&country=in&language=en`
          );
          if (res.ok) {
            const data = await res.json();
            if (data.features && data.features.length > 0) {
              results = data.features.map((f) => ({
                id: f.id,
                name: f.place_name,
                lng: f.center[0],
                lat: f.center[1],
                zoom: f.place_type?.includes("country") ? 5 : f.place_type?.includes("region") ? 8 : 11
              }));
            }
          }
        }

        // 2. Fallback to OpenStreetMap Nominatim for India
        if (results.length === 0) {
          const res = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&countrycodes=in&limit=5`,
            {
              headers: {
                "Accept-Language": "en"
              }
            }
          );
          if (res.ok) {
            const data = await res.json();
            results = data.map((item, idx) => ({
              id: item.place_id || `osm_${idx}`,
              name: item.display_name,
              lat: parseFloat(item.lat),
              lng: parseFloat(item.lon),
              zoom: item.type === "administrative" ? 8 : 11
            }));
          }
        }

        setSuggestions(results);
        setIsOpen(results.length > 0);
      } catch (err) {
        console.error("Geocoding search error:", err);
      } finally {
        setIsLoading(false);
      }
    }, 280);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [query, mapTilerKey]);

  const handleSelect = (item) => {
    setQuery(item.name.split(",")[0]);
    setIsOpen(false);
    setIsSearchExpanded(false);
    if (onSelectLocation) {
      onSelectLocation(item);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && suggestions.length > 0) {
      handleSelect(suggestions[0]);
    }
  };

  return (
    <div className='absolute inset-x-0 top-0 z-50 pointer-events-none'>
      <nav className="h-15 flex justify-between items-center bg-white/95 py-2 px-4 text-black font-sans shadow-[0_2px_4px_rgba(0,0,0,0.1)] pointer-events-auto">
        {/* Brand Name & Logo */}
        <Link 
          to="/" 
          className={`flex items-center gap-2 text-black text-base sm:text-lg md:text-xl font-semibold truncate transition-all duration-200 ${
            isSearchExpanded ? 'hidden sm:flex' : 'flex'
          }`}
        >
          <div className="h-15 rounded-full">
          <img src={logo} alt="GIS Logo" className="h-full w-full object-contain" />
          </div>
          <span>Geospatial Thermal Analysis (GTA)</span>
        </Link>

        {/* Right side search container */}
        <div 
          ref={searchContainerRef}
          className={`relative flex items-center gap-2 sm:gap-4 ml-auto ${
            isSearchExpanded ? 'w-full sm:w-auto' : 'w-auto'
          }`}
        >
          {/* Mobile Search Back Button */}
          {isSearchExpanded && (
            <button 
              onClick={() => {
                setIsSearchExpanded(false);
                setIsOpen(false);
              }}
              className="sm:hidden text-gray-700 p-1 hover:bg-black/10 rounded cursor-pointer"
              aria-label="Close search"
            >
              <X className="w-5 h-5" />
            </button>
          )}

          {/* Search bar container */}
          <div className={`
            border-black
            border-2
            bg-white rounded py-1 px-2.5 h-9 items-center relative transition-all
            ${isSearchExpanded 
              ? 'flex w-full sm:w-72 md:w-88' 
              : 'hidden sm:flex sm:w-72 md:w-88'
            }
          `}>
            <Search className="w-4 h-4 text-gray-500 mr-2 shrink-0" />
            
            <input 
              type="text" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => {
                if (suggestions.length > 0) setIsOpen(true);
              }}
              placeholder="Search location, district, city or lat,lng..." 
              className="border-none outline-none w-full text-xs text-[#333] placeholder-[#888] bg-transparent"
              autoFocus={isSearchExpanded}
            />

            {isLoading && (
              <Loader2 className="w-3.5 h-3.5 text-gray-400 animate-spin shrink-0 mr-1" />
            )}

            {query && (
              <button 
                onClick={() => {
                  setQuery("");
                  setSuggestions([]);
                  setIsOpen(false);
                }}
                className="text-gray-400 hover:text-gray-600 p-0.5 cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}

            {/* Autocomplete Dropdown Suggestions */}
            {isOpen && suggestions.length > 0 && (
              <div className="absolute top-11 left-0 right-0 bg-white border border-gray-200 rounded-lg shadow-2xl overflow-hidden z-50 max-h-64 overflow-y-auto divide-y divide-gray-100">
                {suggestions.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => handleSelect(item)}
                    className="w-full px-3 py-2 text-left hover:bg-gray-50 flex items-start gap-2.5 transition cursor-pointer text-xs"
                  >
                    {item.isCoord ? (
                      <Navigation className="w-4 h-4 text-blue-600 mt-0.5 shrink-0" />
                    ) : (
                      <MapPin className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="font-semibold text-gray-900 truncate">
                        {item.name.split(",")[0]}
                      </p>
                      <p className="text-[10px] text-gray-500 truncate">
                        {item.name}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Search Toggle Button for Mobile */}
          {!isSearchExpanded && (
            <button 
              onClick={() => setIsSearchExpanded(true)}
              className="sm:hidden bg-gray-100 hover:bg-gray-200 text-gray-800 rounded p-1.5 flex items-center justify-center cursor-pointer"
              aria-label="Open search"
            >
              <Search className="w-4 h-4" />
            </button>
          )}

          {/* Info / About Button */}
          <Link 
            to='/about-us' 
            className={`bg-transparent border-none text-black cursor-pointer flex items-center p-0 hover:opacity-80 transition-all ${
              isSearchExpanded ? 'hidden sm:flex' : 'flex'
            }`} 
            aria-label="Information"
          >
            <svg 
              width="24" 
              height="24" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
          </Link>
        </div>
      </nav>
    </div>
  );
};

export default Navbar;