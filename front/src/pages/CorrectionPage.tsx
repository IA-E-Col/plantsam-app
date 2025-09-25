import { useState } from 'react'
import './CorrectionPage.css'

interface CorrectionPageProps {
  images: File[]
  onBack: () => void
}

function CorrectionPage({ images, onBack }: CorrectionPageProps) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [pointType, setPointType] = useState<'positive' | 'negative'>('positive')

  const currentImage = images[currentImageIndex]

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    console.log(`Clicked at: ${x}, ${y} (${pointType} point)`)
    // Ici vous pourrez ajouter la logique pour placer les points plus tard
  }

  const nextImage = () => {
    if (currentImageIndex < images.length - 1) {
      setCurrentImageIndex(currentImageIndex + 1)
    }
  }

  const prevImage = () => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex(currentImageIndex - 1)
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
            <img 
              src={URL.createObjectURL(currentImage)} 
              alt="Original" 
              className="correction-image"
            />
          </div>
          
          <div className="image-container">
            <h3>Segmented Image</h3>
            <img 
              src={URL.createObjectURL(currentImage)} 
              alt="Segmented" 
              className="correction-image"
              onClick={handleImageClick}
            />
            <p>Image {currentImageIndex + 1}/{images.length}</p>
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

          <p className="instruction">Click on the image to correct</p>

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