'use client';

import { useState, useRef } from 'react';
import Image from 'next/image';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResultUrl(null);
    
    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const processImage = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setResultUrl(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      // Using the rewrite rule configured in next.config.ts
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || 'Failed to process image');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setResultUrl(url);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Something went wrong.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-purple-500 selection:text-white overflow-x-hidden">
      
      {/* Background Ambient Glow */}
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-purple-900/20 rounded-full blur-[120px] pointer-events-none z-0" />
      <div className="fixed bottom-0 right-0 w-[800px] h-[600px] bg-blue-900/10 rounded-full blur-[120px] pointer-events-none z-0" />

      <main className="relative z-10 container mx-auto px-4 py-16 max-w-5xl flex flex-col items-center min-h-screen">
        
        {/* Header */}
        <div className="text-center mb-12 space-y-4 animate-in fade-in slide-in-from-top-8 duration-700">
          <div className="inline-block p-px bg-gradient-to-r from-purple-500 to-blue-500 rounded-full mb-4">
             <span className="block px-3 py-1 bg-black/90 rounded-full text-xs font-medium tracking-wider uppercase text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400">
               Next-Gen AI Vision
             </span>
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-white to-white/60">
            Hailo Vision
          </h1>
          <p className="text-lg text-white/40 max-w-xl mx-auto">
            Experience real-time object detection accelerated by the Hailo-8 neural processor. 
            Upload an image to reveal the unseen.
          </p>
        </div>

        {/* Main Interaction Area */}
        <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          
          {/* Upload Section */}
          <div className="space-y-6">
            <div 
              className={`
                relative group cursor-pointer
                border-2 border-dashed rounded-3xl transition-all duration-300 ease-out
                h-[400px] flex flex-col items-center justify-center p-8
                ${isDragging 
                  ? 'border-purple-500 bg-purple-500/10 scale-[1.02]' 
                  : 'border-white/10 hover:border-white/20 hover:bg-white/5 bg-white/[0.02]'
                }
                ${previewUrl ? 'border-solid' : ''}
              `}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => !previewUrl && fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                className="hidden" 
                ref={fileInputRef} 
                onChange={onFileChange} 
                accept="image/*" 
              />

              {previewUrl ? (
                <div className="relative w-full h-full rounded-2xl overflow-hidden group-hover:shadow-2xl transition-all">
                  <Image 
                    src={previewUrl} 
                    alt="Preview" 
                    fill 
                    className="object-cover"
                  />
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setPreviewUrl(null);
                        setSelectedFile(null);
                        setResultUrl(null);
                      }}
                      className="bg-white/10 hover:bg-white/20 text-white px-6 py-3 rounded-full font-medium backdrop-blur-md transition-colors border border-white/10"
                    >
                      Change Image
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center space-y-4 pointer-events-none">
                   <div className="w-20 h-20 bg-gradient-to-tr from-purple-500 to-blue-600 rounded-2xl mx-auto flex items-center justify-center shadow-lg shadow-purple-500/20 group-hover:scale-110 transition-transform duration-300">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10 text-white">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                      </svg>
                   </div>
                   <div>
                    <p className="text-xl font-semibold text-white/80">Upload Image</p>
                    <p className="text-sm text-white/40 mt-1">Drag & Drop or Click to Browse</p>
                   </div>
                </div>
              )}
            </div>

            <button
              onClick={processImage}
              disabled={!selectedFile || isLoading}
              className={`
                w-full py-4 rounded-2xl font-bold text-lg tracking-wide transition-all duration-300
                flex items-center justify-center gap-3
                ${!selectedFile 
                  ? 'bg-white/5 text-white/20 cursor-not-allowed' 
                  : 'bg-white text-black hover:scale-[1.02] shadow-xl shadow-white/10 hover:shadow-white/20'
                }
                ${isLoading ? 'opacity-80 cursor-wait' : ''}
              `}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                <>
                  <span>Generate Analysis</span>
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
                  </svg>
                </>
              )}
            </button>
            
            {error && (
               <div className="p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-sm text-center animate-in fade-in slide-in-from-bottom-2">
                 {error}
               </div>
            )}
          </div>

          {/* Results Section */}
          <div className={`
            relative h-[500px] rounded-3xl overflow-hidden border border-white/10 bg-black/40 backdrop-blur-md
            transition-all duration-500 ease-in-out
            ${resultUrl ? 'shadow-2xl shadow-purple-500/10 ring-1 ring-purple-500/30' : 'opacity-50 grayscale'}
          `}>
             {resultUrl ? (
               <div className="relative w-full h-full group">
                 <Image 
                   src={resultUrl} 
                   alt="Result" 
                   fill 
                   className="object-contain p-4"
                 />
                 <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black via-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="flex items-center justify-between">
                      <span className="text-white font-medium">Detection Complete</span>
                      <a 
                        href={resultUrl} 
                        download="hailo-result.png"
                        className="bg-white text-black px-4 py-2 rounded-lg text-sm font-bold hover:bg-gray-200 transition-colors"
                      >
                        Download
                      </a>
                    </div>
                 </div>
               </div>
             ) : (
               <div className="absolute inset-0 flex flex-col items-center justify-center text-white/20 space-y-4">
                 <div className="w-16 h-16 rounded-full border-2 border-dashed border-white/10 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
                    </svg>
                 </div>
                 <p>Result will appear here</p>
               </div>
             )}
          </div>

        </div>
      </main>
    </div>
  );
}