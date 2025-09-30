import { useState, useRef, useEffect } from 'react'
import './CorrectionPage.css'

interface Point {
  x: number;
  y: number;
  type: 'positive' | 'negative';
  id: number;
}

interface CorrectionPageProps {
  images: File[]
  groupId: string
  onBack: () => void
}

function CorrectionPage({ images, groupId, onBack }: CorrectionPageProps) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [pointType, setPointType] = useState<'positive' | 'negative'>('positive')
  const [points, setPoints] = useState<Point[]>([])
  const [processedImageUrl, setProcessedImageUrl] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const imageRef = useRef<HTMLImageElement>(null)
  const pointIdCounter = useRef(0)

  const currentImage = images[currentImageIndex]

  const loadProcessedImage = async () => {
    if (!groupId) return

    setIsLoading(true)
    try {
      const response = await fetch(`/api/files/group/${groupId}/${currentImageIndex}/result`)
      
      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        setProcessedImageUrl(url)
      } else {
        setProcessedImageUrl(URL.createObjectURL(currentImage))
      }
    } catch (error) {
      console.error('Erreur lors du chargement de l\'image traitée:', error)
      setProcessedImageUrl(URL.createObjectURL(currentImage))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (groupId && images.length > 0) {
      loadProcessedImage()
      setPoints([])
    }
  }, [groupId, currentImageIndex, images.length])

  const applySegmentationWithPoints = async (pointsList: Point[]) => {
    if (!groupId) return

    try {
      const positivePoints = pointsList.filter(p => p.type === 'positive').map(p => [p.x, p.y])
      const negativePoints = pointsList.filter(p => p.type === 'negative').map(p => [p.x, p.y])

      const formData = new FormData()
      formData.append('positivePoints', JSON.stringify(positivePoints))
      formData.append('negativePoints', JSON.stringify(negativePoints))
      
      const response = await fetch(
        `/api/files/group/${groupId}/${currentImageIndex}/segment_with_points`,
        {
          method: 'POST',
          body: formData
        }
      )

      if (response.ok) {
        console.log('Segmentation avec points appliquée avec succès')
        await loadProcessedImage()
      } else {
        console.error('Erreur lors de la segmentation avec points:', response.status)
      }
    } catch (error) {
      console.error('Erreur API:', error)
    }
  }

  const handleImageClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current || !groupId) return

    const img = imageRef.current
    const rect = img.getBoundingClientRect()
    
    const clickX = e.nativeEvent.offsetX
    const clickY = e.nativeEvent.offsetY
    
    console.log(`Clic sur l'image: (${clickX}, ${clickY})`)

    const naturalWidth = img.naturalWidth
    const naturalHeight = img.naturalHeight
    const displayedWidth = rect.width
    const displayedHeight = rect.height
    
    const scaleX = naturalWidth / displayedWidth
    const scaleY = naturalHeight / displayedHeight
    
    const imageX = Math.round(clickX * scaleX)
    const imageY = Math.round(clickY * scaleY)
    
    console.log(`Coordonnées calculées en pixels: (${imageX}, ${imageY})`)

    const displayX = clickX
    const displayY = clickY

    const newPoint: Point = {
      x: displayX,
      y: displayY,
      type: pointType,
      id: pointIdCounter.current++
    }
    
    const newPoints = [...points, newPoint]
    setPoints(newPoints)
    
    await applySegmentationWithPoints(newPoints.map(p => ({
      ...p,
      x: Math.round(p.x * scaleX),
      y: Math.round(p.y * scaleY)
    })))
  }

  const undoLastPoint = async () => {
    if (points.length === 0) return
    
    const newPoints = points.slice(0, -1)
    setPoints(newPoints)
    
    if (newPoints.length === 0) {
      await clearPoints()
    } else {
      if (imageRef.current) {
        const img = imageRef.current
        const rect = img.getBoundingClientRect()
        const naturalWidth = img.naturalWidth
        const naturalHeight = img.naturalHeight
        const displayedWidth = rect.width
        const displayedHeight = rect.height
        const scaleX = naturalWidth / displayedWidth
        const scaleY = naturalHeight / displayedHeight
        
        await applySegmentationWithPoints(newPoints.map(p => ({
          ...p,
          x: Math.round(p.x * scaleX),
          y: Math.round(p.y * scaleY)
        })))
      }
    }
  }

  const clearPoints = async () => {
    try {
      const response = await fetch(
        `/api/files/group/${groupId}/${currentImageIndex}/clear_points`,
        {
          method: 'POST'
        }
      )

      if (response.ok) {
        console.log('Points effacés avec succès')
        setPoints([])
        await loadProcessedImage()
      } else {
        console.error('Erreur lors de l\'effacement des points:', response.status)
      }
    } catch (error) {
      console.error('Erreur API:', error)
    }
  }

  const downloadProcessedImage = async () => {
    if (!processedImageUrl) {
      return
    }

    try {
      const response = await fetch(processedImageUrl)
      const blob = await response.blob()
      
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.style.display = 'none'
      a.href = url
      
      const originalName = images[currentImageIndex].name
      const nameWithoutExtension = originalName.replace(/\.[^/.]+$/, "")
      const extension = 'png'
      a.download = `${nameWithoutExtension}_segmented_${Date.now()}.${extension}`
      
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Erreur lors du téléchargement:', error)
    }
  }

  if (!groupId) {
    return (
      <div className="correction-page">
        <div style={{ padding: '2rem', textAlign: 'center' }}>
          <h2>Erreur: GroupId non défini</h2>
          <button onClick={onBack}>Retour à l'accueil</button>
        </div>
      </div>
    )
  }

  return (
    <div className="correction-page">
      <header className="correction-header">
        <h1>PlantSAM</h1>
        <div>
          <span style={{ marginRight: '1rem', color: '#666' }}>
            Group: {groupId.substring(0, 8)}...
          </span>
          <button className="back-button" onClick={onBack}>
            Back
          </button>
        </div>
      </header>

      <div className="correction-content">
        <div className="images-section">
          <div className="image-container">
            <h3>Segmented Image {isLoading && '(Loading...)'}</h3>
            <div className="image-wrapper">
              <img 
                src={processedImageUrl || URL.createObjectURL(currentImage)} 
                alt="Segmented" 
                className="correction-image"
              />
            </div>
          </div>
          
          <div className="image-container">
            <h3>Original Image</h3>
            <div className="image-wrapper" style={{ position: 'relative' }}>
              <img 
                ref={imageRef}
                src={URL.createObjectURL(currentImage)} 
                alt="Original" 
                className="correction-image clickable-image"
                onClick={handleImageClick}
              />
              
              {points.map(point => (
                <div
                  key={point.id}
                  className={`point ${point.type}-point`}
                  style={{
                    position: 'absolute',
                    left: point.x - 4,
                    top: point.y - 4,
                  }}
                  title={`${point.type} point`}
                />
              ))}
            </div>
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
            <p>Points: {points.length} (✓{points.filter(p => p.type === 'positive').length} ✗{points.filter(p => p.type === 'negative').length})</p>
          </div>

          <div className="action-buttons">
            <button 
              className="control-button undo-button" 
              onClick={undoLastPoint}
              disabled={points.length === 0}
            >
              Undo Last Point
            </button>
            
            <button 
              className="control-button clear-button" 
              onClick={clearPoints}
              disabled={points.length === 0}
            >
              Clear All Points
            </button>

            <button 
              className="control-button download-button" 
              onClick={downloadProcessedImage}
              disabled={!processedImageUrl}
            >
              Download Segmented Image
            </button>
          </div>

          <p className="instruction">
            Click on the original image to place correction points.
          </p>
        </div>
      </div>
    </div>
  )
}

export default CorrectionPage