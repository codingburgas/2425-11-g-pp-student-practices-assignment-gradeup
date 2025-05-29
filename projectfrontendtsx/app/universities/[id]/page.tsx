import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  MapPin,
  Calendar,
  Users,
  GraduationCap,
  Globe,
  Mail,
  Phone,
  BookOpen,
  Building,
  Award,
  ChevronLeft,
} from "lucide-react"
import Link from "next/link"
import Image from "next/image"

// Mock university data
const universities = [
  {
    id: 1,
    name: "Sofia University",
    fullName: 'Sofia University "St. Kliment Ohridski"',
    location: "Sofia, Bulgaria",
    image: "/placeholder.svg?height=400&width=800",
    description:
      "Sofia University is the oldest higher education institution in Bulgaria, founded in 1888. It is consistently ranked as the top university in the country and among the best in Eastern Europe. The university offers a comprehensive range of programs across the humanities, social sciences, natural sciences, and more.",
    longDescription:
      'Sofia University "St. Kliment Ohridski" is the first Bulgarian higher education institution. Its history is an embodiment and a continuation of centuries of cultural and educational tradition in this country. Today it is the largest and most prestigious educational and scientific center in Bulgaria, with 16 faculties and three departments, where about 21,000 students receive their education. The university employs 1,900 highly qualified full-time lecturers and researchers. The university offers 118 degree programs across all major fields of study.',
    founded: 1888,
    students: 25000,
    faculty: 1900,
    programs: 118,
    website: "www.uni-sofia.bg",
    email: "info@uni-sofia.bg",
    phone: "+359 2 9308 200",
    address: "15 Tsar Osvoboditel Blvd, 1504 Sofia, Bulgaria",
    faculties: [
      "Faculty of Biology",
      "Faculty of Chemistry and Pharmacy",
      "Faculty of Classical and Modern Philology",
      "Faculty of Economics and Business Administration",
      "Faculty of Education",
      "Faculty of Geology and Geography",
      "Faculty of History",
      "Faculty of Journalism and Mass Communication",
      "Faculty of Law",
      "Faculty of Mathematics and Informatics",
      "Faculty of Philosophy",
      "Faculty of Physics",
      "Faculty of Pre-School and Primary School Education",
      "Faculty of Slavic Studies",
      "Faculty of Theology",
    ],
    specialties: [
      {
        name: "Computer Science",
        faculty: "Faculty of Mathematics and Informatics",
        degree: "Bachelor",
        duration: "4 years",
      },
      {
        name: "Software Engineering",
        faculty: "Faculty of Mathematics and Informatics",
        degree: "Bachelor",
        duration: "4 years",
      },
      {
        name: "Mathematics",
        faculty: "Faculty of Mathematics and Informatics",
        degree: "Bachelor",
        duration: "4 years",
      },
      { name: "Physics", faculty: "Faculty of Physics", degree: "Bachelor", duration: "4 years" },
      { name: "Chemistry", faculty: "Faculty of Chemistry and Pharmacy", degree: "Bachelor", duration: "4 years" },
      { name: "Biology", faculty: "Faculty of Biology", degree: "Bachelor", duration: "4 years" },
      { name: "Law", faculty: "Faculty of Law", degree: "Master", duration: "5 years" },
      {
        name: "Economics",
        faculty: "Faculty of Economics and Business Administration",
        degree: "Bachelor",
        duration: "4 years",
      },
      {
        name: "Business Administration",
        faculty: "Faculty of Economics and Business Administration",
        degree: "Bachelor",
        duration: "4 years",
      },
    ],
    admissionInfo:
      "Admission to Sofia University is competitive and based on entrance exams specific to each faculty. The application period typically begins in June for the fall semester. International students may have additional requirements, including proof of Bulgarian language proficiency or completion of a preparatory year.",
    facilities: [
      "University Library with over 2 million volumes",
      "Modern computer labs and research facilities",
      "Student dormitories",
      "Sports facilities",
      "Cultural centers and museums",
      "Botanical garden",
    ],
    rankings: [
      { name: "QS World University Rankings", position: "801-1000", year: 2023 },
      { name: "Times Higher Education", position: "1001+", year: 2023 },
      { name: "U.S. News & World Report", position: "1248", year: 2023 },
    ],
  },
]

export default function UniversityDetailsPage({ params }: { params: { id: string } }) {
  const university = universities.find((u) => u.id === Number.parseInt(params.id)) || universities[0]

  return (
    <div className="container mx-auto py-10">
      <div className="mb-6">
        <Button asChild variant="ghost" className="mb-4 gap-1 pl-0 text-muted-foreground">
          <Link href="/universities">
            <ChevronLeft className="h-4 w-4" />
            Back to Universities
          </Link>
        </Button>

        <div className="relative mb-6 h-64 w-full overflow-hidden rounded-xl md:h-80">
          <Image
            src={university.image || "/placeholder.svg"}
            alt={university.name}
            fill
            className="object-cover"
            priority
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
          <div className="absolute bottom-0 left-0 p-6">
            <Badge className="mb-2 bg-[#261FB3]">Est. {university.founded}</Badge>
            <h1 className="text-3xl font-bold text-white md:text-4xl">{university.name}</h1>
            <p className="flex items-center gap-1 text-white/90">
              <MapPin className="h-4 w-4" />
              {university.location}
            </p>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <Card className="border-2 border-[#FBE4D6]">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-[#261FB3]">Students</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center gap-3">
              <Users className="h-8 w-8 text-[#261FB3]" />
              <span className="text-2xl font-bold">{university.students.toLocaleString()}</span>
            </CardContent>
          </Card>

          <Card className="border-2 border-[#FBE4D6]">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-[#261FB3]">Faculty</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center gap-3">
              <GraduationCap className="h-8 w-8 text-[#261FB3]" />
              <span className="text-2xl font-bold">{university.faculty.toLocaleString()}</span>
            </CardContent>
          </Card>

          <Card className="border-2 border-[#FBE4D6]">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-[#261FB3]">Programs</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center gap-3">
              <BookOpen className="h-8 w-8 text-[#261FB3]" />
              <span className="text-2xl font-bold">{university.programs}</span>
            </CardContent>
          </Card>
        </div>
      </div>

      <Tabs defaultValue="overview">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="specialties">Specialties</TabsTrigger>
          <TabsTrigger value="admission">Admission</TabsTrigger>
          <TabsTrigger value="contact">Contact</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>About {university.name}</CardTitle>
              <CardDescription>{university.fullName}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="mb-2 text-lg font-medium">Overview</h3>
                <p className="text-muted-foreground">{university.longDescription}</p>
              </div>

              <div>
                <h3 className="mb-2 text-lg font-medium">Faculties</h3>
                <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-3">
                  {university.faculties.map((faculty) => (
                    <div key={faculty} className="flex items-center gap-2 rounded-md border p-2">
                      <Building className="h-4 w-4 text-[#261FB3]" />
                      <span className="text-sm">{faculty}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="mb-2 text-lg font-medium">Facilities</h3>
                <ul className="grid gap-2 sm:grid-cols-2">
                  {university.facilities.map((facility) => (
                    <li key={facility} className="flex items-center gap-2">
                      <div className="rounded-full bg-[#FBE4D6] p-1">
                        <Award className="h-3 w-3 text-[#261FB3]" />
                      </div>
                      <span className="text-sm">{facility}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="mb-2 text-lg font-medium">Rankings</h3>
                <div className="space-y-2">
                  {university.rankings.map((ranking) => (
                    <div key={ranking.name} className="flex items-center justify-between rounded-md border p-3">
                      <span>{ranking.name}</span>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{ranking.year}</Badge>
                        <Badge className="bg-[#261FB3]">Rank: {ranking.position}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="specialties" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Available Specialties</CardTitle>
              <CardDescription>Browse through the academic programs offered by {university.name}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {university.specialties.map((specialty) => (
                  <div key={specialty.name} className="rounded-lg border p-4">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <h3 className="text-lg font-medium text-[#261FB3]">{specialty.name}</h3>
                      <Badge className="bg-[#261FB3]">{specialty.degree}</Badge>
                    </div>
                    <p className="mt-1 text-sm text-muted-foreground">{specialty.faculty}</p>
                    <div className="mt-2 flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">{specialty.duration}</span>
                    </div>
                    <Button asChild className="mt-4" variant="outline" size="sm">
                      <Link href={`/specialties/${specialty.name.toLowerCase().replace(/\s+/g, "-")}`}>
                        View Details
                      </Link>
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="admission" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Admission Information</CardTitle>
              <CardDescription>Learn about the application process and requirements</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="mb-2 text-lg font-medium">Application Process</h3>
                <p className="text-muted-foreground">{university.admissionInfo}</p>
              </div>

              <div>
                <h3 className="mb-2 text-lg font-medium">Required Documents</h3>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2">
                    <div className="mt-1 rounded-full bg-[#FBE4D6] p-1">
                      <Award className="h-3 w-3 text-[#261FB3]" />
                    </div>
                    <span>Completed application form</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="mt-1 rounded-full bg-[#FBE4D6] p-1">
                      <Award className="h-3 w-3 text-[#261FB3]" />
                    </div>
                    <span>High school diploma or equivalent</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="mt-1 rounded-full bg-[#FBE4D6] p-1">
                      <Award className="h-3 w-3 text-[#261FB3]" />
                    </div>
                    <span>Transcript of records</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="mt-1 rounded-full bg-[#FBE4D6] p-1">
                      <Award className="h-3 w-3 text-[#261FB3]" />
                    </div>
                    <span>Proof of language proficiency (if applicable)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="mt-1 rounded-full bg-[#FBE4D6] p-1">
                      <Award className="h-3 w-3 text-[#261FB3]" />
                    </div>
                    <span>Copy of ID or passport</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="mt-1 rounded-full bg-[#FBE4D6] p-1">
                      <Award className="h-3 w-3 text-[#261FB3]" />
                    </div>
                    <span>Application fee payment receipt</span>
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="mb-2 text-lg font-medium">Important Dates</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between rounded-md border p-3">
                    <span>Application Period Opens</span>
                    <Badge variant="outline">June 1, 2025</Badge>
                  </div>
                  <div className="flex items-center justify-between rounded-md border p-3">
                    <span>Application Deadline</span>
                    <Badge variant="outline">July 15, 2025</Badge>
                  </div>
                  <div className="flex items-center justify-between rounded-md border p-3">
                    <span>Entrance Exams</span>
                    <Badge variant="outline">July 20-30, 2025</Badge>
                  </div>
                  <div className="flex items-center justify-between rounded-md border p-3">
                    <span>Results Announcement</span>
                    <Badge variant="outline">August 10, 2025</Badge>
                  </div>
                  <div className="flex items-center justify-between rounded-md border p-3">
                    <span>Semester Start</span>
                    <Badge variant="outline">September 15, 2025</Badge>
                  </div>
                </div>
              </div>

              <Button className="w-full bg-[#261FB3] hover:bg-[#161179]">Apply Now</Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="contact" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Contact Information</CardTitle>
              <CardDescription>Get in touch with {university.name}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border p-4">
                  <div className="flex items-center gap-3">
                    <div className="rounded-full bg-[#FBE4D6] p-2">
                      <Globe className="h-5 w-5 text-[#261FB3]" />
                    </div>
                    <div>
                      <h3 className="font-medium">Website</h3>
                      <a
                        href={`https://${university.website}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[#261FB3] hover:underline"
                      >
                        {university.website}
                      </a>
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border p-4">
                  <div className="flex items-center gap-3">
                    <div className="rounded-full bg-[#FBE4D6] p-2">
                      <Mail className="h-5 w-5 text-[#261FB3]" />
                    </div>
                    <div>
                      <h3 className="font-medium">Email</h3>
                      <a href={`mailto:${university.email}`} className="text-[#261FB3] hover:underline">
                        {university.email}
                      </a>
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border p-4">
                  <div className="flex items-center gap-3">
                    <div className="rounded-full bg-[#FBE4D6] p-2">
                      <Phone className="h-5 w-5 text-[#261FB3]" />
                    </div>
                    <div>
                      <h3 className="font-medium">Phone</h3>
                      <a href={`tel:${university.phone}`} className="text-[#261FB3] hover:underline">
                        {university.phone}
                      </a>
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border p-4">
                  <div className="flex items-center gap-3">
                    <div className="rounded-full bg-[#FBE4D6] p-2">
                      <MapPin className="h-5 w-5 text-[#261FB3]" />
                    </div>
                    <div>
                      <h3 className="font-medium">Address</h3>
                      <p className="text-muted-foreground">{university.address}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border p-4">
                <h3 className="mb-4 text-lg font-medium">Location Map</h3>
                <div className="relative h-64 w-full overflow-hidden rounded-md bg-muted">
                  <Image
                    src="/placeholder.svg?height=300&width=600"
                    alt="University location map"
                    fill
                    className="object-cover"
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <p className="rounded bg-background/80 p-2 text-sm">Map placeholder</p>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border p-4">
                <h3 className="mb-4 text-lg font-medium">Contact Form</h3>
                <form className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label htmlFor="name" className="text-sm font-medium">
                        Name
                      </label>
                      <input
                        id="name"
                        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                        placeholder="Your name"
                      />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="email" className="text-sm font-medium">
                        Email
                      </label>
                      <input
                        id="email"
                        type="email"
                        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                        placeholder="Your email"
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label htmlFor="subject" className="text-sm font-medium">
                      Subject
                    </label>
                    <input
                      id="subject"
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                      placeholder="Message subject"
                    />
                  </div>
                  <div className="space-y-2">
                    <label htmlFor="message" className="text-sm font-medium">
                      Message
                    </label>
                    <textarea
                      id="message"
                      rows={4}
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                      placeholder="Your message"
                    ></textarea>
                  </div>
                  <Button className="w-full bg-[#261FB3] hover:bg-[#161179]">Send Message</Button>
                </form>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
