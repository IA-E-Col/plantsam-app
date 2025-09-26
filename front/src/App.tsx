import { useState } from 'react'
import './App.css'
import HomePage from './components/HomePage'
import CorrectionPage from './components/CorrectionPage'

type Page = 'home' | 'correction'

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home')
  const [selectedImages, setSelectedImages] = useState<File[]>([])
  const [groupId, setGroupId] = useState<string>('')

  const handleImagesSelected = (images: File[], newGroupId: string) => {
    console.log('App: Images selected with groupId:', newGroupId)
    setSelectedImages(images)
    setGroupId(newGroupId)
    setCurrentPage('correction')
  }

  const handleBack = () => {
    setCurrentPage('home')
    setSelectedImages([])
    setGroupId('')
  }

  return (
    <div className="app">
      {currentPage === 'home' ? (
        <HomePage 
          onImagesSelected={handleImagesSelected}
        />
      ) : (
        <CorrectionPage 
          images={selectedImages}
          groupId={groupId}
          onBack={handleBack}
        />
      )}
    </div>
  )
}

export default App
