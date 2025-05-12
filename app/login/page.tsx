"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export default function LoginPage() {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const router = useRouter()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // En una aplicación real, aquí iría la autenticación
    // Por ahora, simplemente redirigimos al dashboard
    router.push("/dashboard")
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-planck-beige p-4">
      <Link href="/" className="absolute left-4 top-4 flex items-center gap-2 md:left-8 md:top-8">
        <div className="relative h-8 w-8">
          <Image src="/logo.png" alt="Planck Logo" fill className="object-contain" />
        </div>
        <span className="font-bold text-planck-dark">Planck</span>
      </Link>

      <Card className="w-full max-w-md bg-white border-none">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold text-center text-planck-dark">Iniciar Sesión</CardTitle>
          <CardDescription className="text-center">Ingrese sus credenciales para acceder al sistema</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Usuario</Label>
              <Input
                id="username"
                placeholder="Ingrese su nombre de usuario"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="border-planck-teal/20"
              />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="password">Contraseña</Label>
                <Link href="/forgot-password" className="text-xs text-planck-teal hover:underline">
                  ¿Olvidó su contraseña?
                </Link>
              </div>
              <Input
                id="password"
                type="password"
                placeholder="Ingrese su contraseña"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="border-planck-teal/20"
              />
            </div>
            <Button type="submit" className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white">
              Ingresar
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex flex-col">
          <div className="text-center text-sm text-muted-foreground mt-2">
            ¿No tiene una cuenta?{" "}
            <Link href="/register" className="text-planck-teal hover:underline">
              Registrarse
            </Link>
          </div>
        </CardFooter>
      </Card>
    </div>
  )
}
