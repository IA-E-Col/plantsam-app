import { useState, useRef } from 'react'
import './CorrectionPage.css'

interface Point {
  x: number;
  y: number;
  type: 'positive' | 'negative';
  id: number;
}

interface CorrectionPageProps {
  images: File[]
  onBack: () => void
}

function CorrectionPage({ images, onBack }: CorrectionPageProps) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [pointType, setPointType] = useState<'positive' | 'negative'>('positive')
  const [points, setPoints] = useState<Point[]>([])
  const imageRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const pointIdCounter = useRef(0)

  const currentImage = images[currentImageIndex]

  const handleImageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return

    const container = containerRef.current
    const rect = container.getBoundingClientRect()
    
    // Calculer les coordonnées du clic par rapport au conteneur
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    // Vérifier que le clic est bien dans le conteneur
    if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
      const newPoint: Point = {
        x,
        y,
        type: pointType,
        id: pointIdCounter.current++
      }
      
      setPoints(prevPoints => [...prevPoints, newPoint])
    }
  }

  const clearPoints = () => {
    setPoints([])
  }

  const savePoints = () => {
    // Préparer les données à sauvegarder
    const saveData = {
      imageIndex: currentImageIndex,
      imageName: images[currentImageIndex].name,
      points: points,
      timestamp: new Date().toISOString()
    }
    
    // Pour l'instant, on affiche juste les données dans la console
    console.log('Données à sauvegarder:', saveData)
    
    // Ici vous pourrez ajouter la logique pour envoyer les données au serveur
    alert(`Points sauvegardés pour l'image ${currentImageIndex + 1}! (Voir la console pour les détails)`)
  }

  const nextImage = () => {
    if (currentImageIndex < images.length - 1) {
      setCurrentImageIndex(currentImageIndex + 1)
      setPoints([])
    }
  }

  const prevImage = () => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex(currentImageIndex - 1)
      setPoints([])
    }
  }

  return (
    <div className="correction-page">
      <header className="correction-header">
        <h1>PlantSAM</h1>
        <button className="back-button" onClick={onBack}>
          Back
        </button>
      </header>

      <div className="correction-content">
        <div className="images-section">
          <div className="image-container">
            <h3>Original Image</h3>
            <div className="image-wrapper">
              <img 
                src={URL.createObjectURL(currentImage)} 
                alt="Original" 
                className="correction-image"
              />
            </div>
          </div>
          
          <div className="image-container">
            <h3>Segmented Image</h3>
            <div 
              ref={containerRef}
              className="image-wrapper" 
              onClick={handleImageClick}
              style={{ position: 'relative', cursor: 'crosshair' }}
            >
              <img 
                src={URL.createObjectURL(currentImage)} 
                alt="Segmented" 
                className="correction-image"
                style={{ pointerEvents: 'none' }}
              />
              
              {/* Points placés sur l'image */}
              {points.map(point => (
                <div
                  key={point.id}
                  className={`point ${point.type}-point`}
                  style={{
                    position: 'absolute',
                    left: point.x - 6,
                    top: point.y - 6,
                  }}
                />
              ))}
            </div>
            <p>Image {currentImageIndex + 1} of {images.length}</p>
          </div>
        </div>

        <div className="correction-controls">
          <h3>Correction</h3>
          
          <div className="point-selector">
            <label>
              <input
                type="radio"
                name="pointType"
                value="positive"
                checked={pointType === 'positive'}
                onChange={() => setPointType('positive')}
              />
              Positive point
            </label>
            
            <label>
              <input
                type="radio"
                name="pointType"
                value="negative"
                checked={pointType === 'negative'}
                onChange={() => setPointType('negative')}
              />
              Negative point
            </label>
          </div>

          <div className="points-info">
            <p>Points placed: {points.length}</p>
            <button className="clear-button" onClick={clearPoints}>
              Clear Points
            </button>
          </div>

          <p className="instruction">Click on the segmented image to place correction points</p>

          <div className="save-section">
            <button 
              className="save-button" 
              onClick={savePoints}
              disabled={points.length === 0}
            >
              Save Points
            </button>
          </div>

          <div className="navigation-buttons">
            <button onClick={prevImage} disabled={currentImageIndex === 0}>
              Previous
            </button>
            <button onClick={nextImage} disabled={currentImageIndex === images.length - 1}>
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CorrectionPage