import { useState } from 'react'
import './App.css'
import HomePage from './components/HomePage'
import CorrectionPage from './components/CorrectionPage'

type Page = 'home' | 'correction'

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home')
  const [selectedImages, setSelectedImages] = useState<File[]>([])

  return (
    <div className="app">
      {currentPage === 'home' ? (
        <HomePage 
          onImagesSelected={setSelectedImages}
          onProceed={() => setCurrentPage('correction')}
        />
      ) : (
        <CorrectionPage 
          images={selectedImages}
          onBack={() => setCurrentPage('home')}
        />
      )}
    </div>
  )
}

export default App