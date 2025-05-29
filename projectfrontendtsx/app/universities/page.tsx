import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Search, MapPin, BookOpen, Users, GraduationCap } from "lucide-react"
import Link from "next/link"
import Image from "next/image"

// Mock university data
const universities = [
  {
    id: 1,
    name: "Sofia University",
    location: "Sofia",
    image: "/placeholder.svg?height=200&width=300",
    description: "The oldest higher education institution in Bulgaria, offering a wide range of programs.",
    programs: 78,
    students: 25000,
    founded: 1888,
    specialties: ["Computer Science", "Mathematics", "Physics", "Law", "Economics"],
  },
  {
    id: 2,
    name: "Technical University of Sofia",
    location: "Sofia",
    image: "/placeholder.svg?height=200&width=300",
    description: "The largest technical university in Bulgaria, specializing in engineering and technology.",
    programs: 42,
    students: 18000,
    founded: 1945,
    specialties: ["Mechanical Engineering", "Electrical Engineering", "Computer Science", "Telecommunications"],
  },
  {
    id: 3,
    name: "American University in Bulgaria",
    location: "Blagoevgrad",
    image: "/placeholder.svg?height=200&width=300",
    description: "A private university offering American-style liberal arts education in Bulgaria.",
    programs: 13,
    students: 1000,
    founded: 1991,
    specialties: ["Business Administration", "Economics", "Computer Science", "Political Science"],
  },
  {
    id: 4,
    name: "Medical University of Sofia",
    location: "Sofia",
    image: "/placeholder.svg?height=200&width=300",
    description: "The oldest and largest medical university in Bulgaria.",
    programs: 15,
    students: 8000,
    founded: 1917,
    specialties: ["Medicine", "Dentistry", "Pharmacy", "Public Health"],
  },
  {
    id: 5,
    name: "New Bulgarian University",
    location: "Sofia",
    image: "/placeholder.svg?height=200&width=300",
    description: "A private university known for its innovative approach to education.",
    programs: 50,
    students: 12000,
    founded: 1991,
    specialties: ["Arts", "Design", "Psychology", "Mass Communication", "Law"],
  },
  {
    id: 6,
    name: "University of National and World Economy",
    location: "Sofia",
    image: "/placeholder.svg?height=200&width=300",
    description:
      "The largest economics university in Bulgaria, offering programs in economics, finance, and management.",
    programs: 45,
    students: 20000,
    founded: 1920,
    specialties: ["Economics", "Finance", "Management", "Marketing", "International Relations"],
  },
]

export default function UniversitiesPage() {
  return (
    <div className="container mx-auto py-10">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-[#261FB3] md:text-4xl">Universities Catalog</h1>
        <p className="mt-2 text-muted-foreground">
          Explore universities and find the perfect match for your academic journey
        </p>
      </div>

      <div className="mb-8 flex flex-col gap-4 sm:flex-row">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input placeholder="Search universities..." className="pl-9" />
        </div>
        <div className="flex gap-2">
          <Button variant="outline">Filter</Button>
          <Button variant="outline">Sort</Button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {universities.map((university) => (
          <Card
            key={university.id}
            className="overflow-hidden border-2 border-[#FBE4D6] transition-all hover:shadow-lg"
          >
            <div className="relative h-48 w-full">
              <Image src={university.image || "/placeholder.svg"} alt={university.name} fill className="object-cover" />
            </div>
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between">
                <CardTitle className="text-xl text-[#261FB3]">{university.name}</CardTitle>
                <Badge className="bg-[#261FB3]">Est. {university.founded}</Badge>
              </div>
              <CardDescription className="flex items-center gap-1">
                <MapPin className="h-3.5 w-3.5" />
                {university.location}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground line-clamp-3">{university.description}</p>
              <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                <div className="rounded-md bg-[#FBE4D6] p-2">
                  <BookOpen className="mx-auto h-4 w-4 text-[#261FB3]" />
                  <p className="mt-1 text-xs font-medium">{university.programs} Programs</p>
                </div>
                <div className="rounded-md bg-[#FBE4D6] p-2">
                  <Users className="mx-auto h-4 w-4 text-[#261FB3]" />
                  <p className="mt-1 text-xs font-medium">{university.students.toLocaleString()} Students</p>
                </div>
                <div className="rounded-md bg-[#FBE4D6] p-2">
                  <GraduationCap className="mx-auto h-4 w-4 text-[#261FB3]" />
                  <p className="mt-1 text-xs font-medium">{university.specialties.length} Specialties</p>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-1">
                {university.specialties.slice(0, 3).map((specialty) => (
                  <Badge key={specialty} variant="outline" className="text-xs">
                    {specialty}
                  </Badge>
                ))}
                {university.specialties.length > 3 && (
                  <Badge variant="outline" className="text-xs">
                    +{university.specialties.length - 3} more
                  </Badge>
                )}
              </div>
            </CardContent>
            <CardFooter>
              <Button asChild className="w-full bg-[#261FB3] hover:bg-[#161179]">
                <Link href={`/universities/${university.id}`}>View Details</Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  )
}
