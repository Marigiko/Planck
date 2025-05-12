"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Upload, Check, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface DataUploaderProps {
  onUploadComplete: () => void
}

export function DataUploader({ onUploadComplete }: DataUploaderProps) {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      if (selectedFile.name.endsWith(".csv")) {
        setFile(selectedFile)
        setError(null)
      } else {
        setFile(null)
        setError("Por favor, seleccione un archivo CSV válido.")
      }
    }
  }

  const handleUpload = () => {
    if (!file) {
      setError("Por favor, seleccione un archivo para cargar.")
      return
    }

    setUploading(true)
    setProgress(0)
    setError(null)

    // Simulamos la carga del archivo
    const interval = setInterval(() => {
      setProgress((prevProgress) => {
        if (prevProgress >= 100) {
          clearInterval(interval)
          setUploading(false)
          setSuccess(true)
          setTimeout(() => {
            onUploadComplete()
          }, 1000)
          return 100
        }
        return prevProgress + 10
      })
    }, 300)
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="file">Archivo CSV</Label>
        <Input
          id="file"
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          disabled={uploading}
          className="border-planck-teal/20"
        />
        <p className="text-xs text-muted-foreground">
          Seleccione un archivo CSV con sus datos. El archivo debe contener características y etiquetas.
        </p>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="bg-green-50 text-green-800 border-green-200">
          <Check className="h-4 w-4" />
          <AlertTitle>Éxito</AlertTitle>
          <AlertDescription>Archivo cargado correctamente.</AlertDescription>
        </Alert>
      )}

      {uploading && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Cargando archivo...</span>
            <span>{progress}%</span>
          </div>
          <Progress value={progress} className="h-2 bg-planck-beige [&>div]:bg-planck-teal" />
        </div>
      )}

      <Button
        onClick={handleUpload}
        disabled={!file || uploading || success}
        className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white"
      >
        {uploading ? (
          "Cargando..."
        ) : (
          <>
            <Upload className="mr-2 h-4 w-4" /> Cargar Archivo
          </>
        )}
      </Button>
    </div>
  )
}
