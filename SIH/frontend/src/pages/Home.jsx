import React, { useRef, useState } from 'react'
import Navbar from '../components/Navbar'
import Map from '../components/Map'
import Sidebar from '../components/Sidebar'
import TimelineControl from '../components/TimelineControl'

import gsap from 'gsap'
import { useGSAP } from '@gsap/react'

const API_BASE_URL = 'http://localhost:8000/api/v1'

const Home = () => {
  const [firePannel, setFirePannel] = useState(false)
  const [selectedHotspot, setSelectedHotspot] = useState(null)
  const [predictionData, setPredictionData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Timeline and Filter state
  const [activePreset, setActivePreset] = useState('live')
  const [activeFilter, setActiveFilter] = useState('ALL')
  const [totalMatches, setTotalMatches] = useState(0)
  const [displayedCount, setDisplayedCount] = useState(0)
  const [isLoadingHotspots, setIsLoadingHotspots] = useState(true)

  const firePannelRef = useRef(null)

  useGSAP(() => {
    const mob = window.innerWidth < 768
    if (firePannel) {
      gsap.to(firePannelRef.current, {
        x: 0,
        y: 0,
        opacity: 1,
        duration: 0.35,
        ease: 'power2.out'
      })
    } else {
      gsap.to(firePannelRef.current, {
        x: mob ? 0 : '100%',
        y: mob ? '100%' : 0,
        opacity: 0,
        duration: 0.3,
        ease: 'power2.in'
      })
    }
  }, [firePannel])

  const handleSelectHotspot = async (hotspot) => {
    setSelectedHotspot(hotspot)
    setFirePannel(true)
    setIsLoading(true)
    setError(null)

    try {
      const payload = {
        event_id: hotspot.event_id || '',
        latitude: parseFloat(hotspot.latitude),
        longitude: parseFloat(hotspot.longitude),
        frp: parseFloat(hotspot.frp || hotspot.avg_frp || 10.0),
        bright_ti4: parseFloat(hotspot.bright_ti4 || 330.0),
        bright_ti5: parseFloat(hotspot.bright_ti5 || 300.0),
        daynight: hotspot.daynight || 'D',
        acq_date: hotspot.acq_date || '',
        acq_time: hotspot.acq_time || '',
        satellite: hotspot.satellite || 'VIIRS'
      }

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`)
      }

      const data = await response.json()
      setPredictionData(data)
    } catch (err) {
      console.error('Prediction API error:', err)
      setError('Failed to load fire classification. Ensure FastAPI backend is running.')
    } finally {
      setIsLoading(false)
    }
  }

  // Search location state
  const [searchLocation, setSearchLocation] = useState(null)

  const handleSelectLocation = (location) => {
    setSearchLocation(location)
  }

  return (
    <div className='relative w-screen h-screen overflow-hidden bg-gray-100'>
      <Navbar onSelectLocation={handleSelectLocation} />
      
      {/* Interactive Map */}
      <Map
        onSelectHotspot={handleSelectHotspot}
        searchLocation={searchLocation}
        activePreset={activePreset}
        activeFilter={activeFilter}
        setTotalMatches={setTotalMatches}
        setDisplayedCount={setDisplayedCount}
        setIsLoadingHotspots={setIsLoadingHotspots}
        isLoadingHotspots={isLoadingHotspots}
      />

      {/* NASA FIRMS Timeline Control Bar */}
      <TimelineControl
        activePreset={activePreset}
        setActivePreset={setActivePreset}
        activeFilter={activeFilter}
        setActiveFilter={setActiveFilter}
        totalMatches={totalMatches}
        displayedCount={displayedCount}
        isLoading={isLoadingHotspots}
      />

      {/* Slide-in Analysis Sidebar */}
      <div
        ref={firePannelRef}
        className='fixed bottom-2 right-2 translate-y-full md:translate-x-full md:top-20 md:right-2 w-95/100 md:w-96 z-40 max-h-[85vh] overflow-y-auto'
      >
        <Sidebar
          setFirePannel={setFirePannel}
          hotspot={selectedHotspot}
          prediction={predictionData}
          isLoading={isLoading}
          error={error}
        />
      </div>
    </div>
  )
}

export default Home