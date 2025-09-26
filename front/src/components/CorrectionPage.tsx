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
    }
  }, [groupId, currentImageIndex, images.length])

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
    const scale = Math.max(scaleX, scaleY)
    
    const imageX = clickX * scale
    const imageY = clickY * scale
    
    const xPercent = (imageX / naturalWidth) * 100
    const yPercent = (imageY / naturalHeight) * 100
    
    console.log(`Coordonnées calculées: Image (${imageX}, ${imageY}) -> Pourcentage: (${xPercent}%, ${yPercent}%)`)

    const container = img.parentElement
    if (!container) return
    
    const containerRect = container.getBoundingClientRect()
    const xInContainer = clickX + (rect.left - containerRect.left)
    const yInContainer = clickY + (rect.top - containerRect.top)

    const newPoint: Point = {
      x: xInContainer,
      y: yInContainer,
      type: pointType,
      id: pointIdCounter.current++
    }
    
    setPoints(prevPoints => [...prevPoints, newPoint])

    try {
      const endpoint = pointType === 'positive' ? 'positive' : 'negative'
      
      const response = await fetch(
        `/api/files/group/${groupId}/${currentImageIndex}/point/${endpoint}?x=${Math.round(xPercent)}&y=${Math.round(yPercent)}`,
        {
          method: 'POST'
        }
      )

      if (response.ok) {
        console.log(`Point ${pointType} ajouté avec succès`)
        await loadProcessedImage()
      } else {
        console.error('Erreur lors de l\'ajout du point:', response.status)
      }
    } catch (error) {
      console.error('Erreur API:', error)
    }
  }

  const clearPoints = () => {
    setPoints([])
    loadProcessedImage()
  }

  const downloadProcessedImage = async () => {
    if (!processedImageUrl) {
      alert('Aucune image traitée à télécharger')
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
      
      alert('Image téléchargée avec succès!')
    } catch (error) {
      console.error('Erreur lors du téléchargement:', error)
      alert('Erreur lors du téléchargement de l\'image')
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
                    left: point.x - 6,
                    top: point.y - 6,
                  }}
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
            <p>Points placed: {points.length}</p>
            <button className="clear-button" onClick={clearPoints}>
              Clear Points
            </button>
          </div>

          <p className="instruction">Click on the original image to place correction points</p>

          <div className="save-section">
            <button 
              className="save-button" 
              onClick={downloadProcessedImage}
            >
              Download Segmented Image
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CorrectionPage
