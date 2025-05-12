import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-planck-beige">
      <header className="sticky top-0 z-50 w-full border-b border-planck-beige/20 bg-planck-beige/95 backdrop-blur supports-[backdrop-filter]:bg-planck-beige/60">
        <div className="container flex h-16 items-center">
          <div className="flex items-center gap-2">
            <div className="relative h-8 w-8">
              <Image src="/logo.png" alt="Planck Logo" fill className="object-contain" />
            </div>
            <span className="text-xl font-bold text-planck-dark">Planck</span>
          </div>
          <nav className="ml-auto flex gap-4">
            <Link href="/" className="text-sm font-medium text-planck-dark">
              Inicio
            </Link>
            <Link href="/login" className="text-sm font-medium text-planck-dark">
              Iniciar Sesión
            </Link>
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32">
          <div className="container px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-2 lg:gap-12 items-center">
              <div className="space-y-4">
                <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl text-planck-dark">
                  Effortless Quantum Solutions
                </h1>
                <p className="max-w-[600px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  Connect your data and start using quantum computing effortlessly, AI-enhanced.
                </p>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Link href="/login">
                    <Button size="lg" className="bg-planck-teal hover:bg-planck-teal/90 text-white">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="mr-2 h-5 w-5"
                      >
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                      </svg>
                      Opening soon
                    </Button>
                  </Link>
                </div>
              </div>
              <div className="mx-auto lg:mx-0 lg:flex lg:justify-center">
                <div className="relative h-[300px] w-[300px] sm:h-[400px] sm:w-[400px] lg:h-[500px] lg:w-[500px]">
                  <Image
                    src="/placeholder.svg?height=500&width=500"
                    alt="Visualización Cuántica"
                    fill
                    className="object-contain"
                  />
                </div>
              </div>
            </div>
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-32 bg-white">
          <div className="container px-4 md:px-6">
            <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl text-planck-dark">
                Algoritmos Cuánticos Disponibles
              </h2>
              <p className="max-w-[85%] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Utilice nuestros algoritmos cuánticos para mejorar sus modelos de aprendizaje automático.
              </p>
            </div>
            <div className="mx-auto grid max-w-5xl gap-6 py-12 lg:grid-cols-3">
              <Card className="bg-planck-beige border-none">
                <CardHeader>
                  <CardTitle className="text-planck-dark">QSVM</CardTitle>
                  <CardDescription>Quantum Support Vector Machine</CardDescription>
                </CardHeader>
                <CardContent>
                  <p>
                    Utiliza un mapa de características cuántico para calcular un kernel cuántico y entrenar un
                    clasificador SVM.
                  </p>
                </CardContent>
                <CardFooter>
                  <Link href="/login" className="w-full">
                    <Button className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white">Probar QSVM</Button>
                  </Link>
                </CardFooter>
              </Card>
              <Card className="bg-planck-beige border-none">
                <CardHeader>
                  <CardTitle className="text-planck-dark">QPCA</CardTitle>
                  <CardDescription>Quantum Principal Component Analysis</CardDescription>
                </CardHeader>
                <CardContent>
                  <p>Convierte los datos clásicos a un formato cuántico y aplica PCA para reducir dimensiones.</p>
                </CardContent>
                <CardFooter>
                  <Link href="/login" className="w-full">
                    <Button className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white">Probar QPCA</Button>
                  </Link>
                </CardFooter>
              </Card>
              <Card className="bg-planck-beige border-none">
                <CardHeader>
                  <CardTitle className="text-planck-dark">QAOA</CardTitle>
                  <CardDescription>Quantum Approximate Optimization Algorithm</CardDescription>
                </CardHeader>
                <CardContent>
                  <p>
                    Resuelve problemas de Max-Cut mediante un circuito QAOA, optimizando la función de costo con COBYLA.
                  </p>
                </CardContent>
                <CardFooter>
                  <Link href="/login" className="w-full">
                    <Button className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white">Probar QAOA</Button>
                  </Link>
                </CardFooter>
              </Card>
            </div>
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-32 bg-planck-beige">
          <div className="container px-4 md:px-6">
            <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl text-planck-dark">Pricing</h2>
              <p className="max-w-[85%] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Choose the plan that fits your needs
              </p>
            </div>
            <div className="mx-auto grid max-w-5xl gap-6 py-12 lg:grid-cols-3">
              <Card className="bg-planck-beige border border-planck-teal/20">
                <CardHeader className="text-center">
                  <CardTitle className="text-planck-dark">Basic plan</CardTitle>
                  <div className="text-4xl font-bold text-planck-teal mt-2">Free</div>
                  <CardDescription>Best for starters</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Quantum simulators</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Low execution volume</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Free simulated executions</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Basic benchmarks and tools</span>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4"
                    >
                      <path d="M5 12h14"></path>
                      <path d="M12 5v14"></path>
                    </svg>
                    Opening soon
                  </Button>
                </CardFooter>
              </Card>
              <Card className="bg-planck-beige border border-planck-teal/20">
                <CardHeader className="text-center">
                  <CardTitle className="text-planck-dark">Growth plan</CardTitle>
                  <div className="text-4xl font-bold text-planck-teal mt-2">$19/month</div>
                  <CardDescription>Best for low volume</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Quantum computers</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Medium execution volume</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>0.00007 USD / execution</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Advanced benchmarks and tools</span>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4"
                    >
                      <path d="M5 12h14"></path>
                      <path d="M12 5v14"></path>
                    </svg>
                    Opening soon
                  </Button>
                </CardFooter>
              </Card>
              <Card className="bg-planck-beige border border-planck-teal/20">
                <CardHeader className="text-center">
                  <CardTitle className="text-planck-dark">Enterprise plan</CardTitle>
                  <div className="text-4xl font-bold text-planck-teal mt-2">$199/month</div>
                  <CardDescription>Best for high volume</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Quantum computers</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Large execution volume</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>0.00006 USD / execution</span>
                  </div>
                  <div className="flex items-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mr-2 h-4 w-4 text-planck-teal"
                    >
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Custom benchmarks and tools</span>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full bg-planck-teal hover:bg-planck-teal/90 text-white">Contact Us</Button>
                </CardFooter>
              </Card>
            </div>
            <div className="text-center text-sm text-muted-foreground mt-4">
              Executions on the Basic plan are executed with quantum inspired algorithms, it helps you understand the
              basics before upgrading.
              <br />
              Billed in packs of 50.000 executions.
            </div>
          </div>
        </section>
      </main>
      <footer className="border-t py-6 md:py-0 bg-planck-beige">
        <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
          <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
            © 2024 Planck. Todos los derechos reservados.
          </p>
          <div className="flex gap-4">
            <Link href="/about" className="text-sm font-medium text-planck-dark">
              Acerca de
            </Link>
            <Link href="/contact" className="text-sm font-medium text-planck-dark">
              Contacto
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
