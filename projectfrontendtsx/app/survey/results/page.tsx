"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Link from "next/link"
import { ChartContainer, ChartBars, ChartBar } from "@/components/ui/chart"
import { School, BookOpen, GraduationCap, Download, Share2 } from "lucide-react"

// Mock recommendation data
const recommendationData = {
  topSpecialties: [
    {
      id: 1,
      name: "Computer Science",
      score: 92,
      description: "Based on your strong analytical skills and interest in technology",
    },
    {
      id: 2,
      name: "Business Administration",
      score: 85,
      description: "Matches your communication skills and interest in business",
    },
    { id: 3, name: "Psychology", score: 78, description: "Aligns with your interest in understanding human behavior" },
  ],
  topUniversities: [
    {
      id: 1,
      name: "Sofia University",
      score: 95,
      location: "Sofia",
      programs: ["Computer Science", "Business Administration"],
    },
    {
      id: 2,
      name: "Technical University of Sofia",
      score: 88,
      location: "Sofia",
      programs: ["Computer Science", "Engineering"],
    },
    {
      id: 3,
      name: "American University in Bulgaria",
      score: 82,
      location: "Blagoevgrad",
      programs: ["Business Administration", "Economics"],
    },
  ],
  skillsAnalysis: {
    analytical: 85,
    creative: 65,
    communication: 75,
    teamwork: 80,
    problemSolving: 90,
  },
}

export default function SurveyResultsPage() {
  return (
    <div className="container mx-auto py-10">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-[#261FB3] md:text-4xl">Your Personalized Results</h1>
        <p className="mt-2 text-muted-foreground">
          Based on your survey responses, we've generated personalized recommendations for you
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card className="border-2 border-[#FBE4D6] shadow-lg">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-[#261FB3]">
              <BookOpen className="h-5 w-5" />
              Match Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center py-6">
              <div className="relative flex h-40 w-40 items-center justify-center rounded-full border-8 border-[#FBE4D6] text-center">
                <div className="text-4xl font-bold text-[#261FB3]">92%</div>
              </div>
              <p className="mt-4 text-center text-sm text-muted-foreground">
                Your responses show a strong match with your recommended specialties
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-2 border-[#FBE4D6] shadow-lg">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-[#261FB3]">
              <School className="h-5 w-5" />
              Top University
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center py-6">
              <div className="flex h-40 w-40 items-center justify-center">
                <div className="text-center">
                  <div className="text-2xl font-bold text-[#261FB3]">Sofia University</div>
                  <div className="mt-2 text-sm text-muted-foreground">Sofia, Bulgaria</div>
                  <Badge className="mt-3 bg-[#261FB3]">95% Match</Badge>
                </div>
              </div>
              <Button asChild className="mt-4 bg-[#261FB3] hover:bg-[#161179]">
                <Link href="/universities/1">View Details</Link>
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="border-2 border-[#FBE4D6] shadow-lg">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-[#261FB3]">
              <GraduationCap className="h-5 w-5" />
              Top Specialty
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center py-6">
              <div className="flex h-40 w-40 items-center justify-center">
                <div className="text-center">
                  <div className="text-2xl font-bold text-[#261FB3]">Computer Science</div>
                  <div className="mt-2 text-sm text-muted-foreground">Technology & IT</div>
                  <Badge className="mt-3 bg-[#261FB3]">92% Match</Badge>
                </div>
              </div>
              <Button asChild className="mt-4 bg-[#261FB3] hover:bg-[#161179]">
                <Link href="/specialties/1">View Details</Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="mt-8">
        <Tabs defaultValue="specialties">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="specialties">Recommended Specialties</TabsTrigger>
            <TabsTrigger value="universities">Recommended Universities</TabsTrigger>
            <TabsTrigger value="skills">Skills Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="specialties" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Top Specialty Recommendations</CardTitle>
                <CardDescription>Based on your interests, skills, and career goals</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {recommendationData.topSpecialties.map((specialty) => (
                    <div key={specialty.id} className="rounded-lg border p-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-medium">{specialty.name}</h3>
                        <Badge className="bg-[#261FB3]">{specialty.score}% Match</Badge>
                      </div>
                      <p className="mt-2 text-sm text-muted-foreground">{specialty.description}</p>
                      <Button asChild className="mt-4" variant="outline" size="sm">
                        <Link href={`/specialties/${specialty.id}`}>Learn More</Link>
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="universities" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Top University Recommendations</CardTitle>
                <CardDescription>Institutions that offer your recommended specialties</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {recommendationData.topUniversities.map((university) => (
                    <div key={university.id} className="rounded-lg border p-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-medium">{university.name}</h3>
                        <Badge className="bg-[#261FB3]">{university.score}% Match</Badge>
                      </div>
                      <p className="mt-1 text-sm text-muted-foreground">{university.location}</p>
                      <div className="mt-2">
                        <span className="text-sm font-medium">Programs: </span>
                        <span className="text-sm text-muted-foreground">{university.programs.join(", ")}</span>
                      </div>
                      <Button asChild className="mt-4" variant="outline" size="sm">
                        <Link href={`/universities/${university.id}`}>View University</Link>
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="skills" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Your Skills Analysis</CardTitle>
                <CardDescription>Based on your self-assessment and survey responses</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ChartContainer
                    title="Skills Assessment"
                    description="Your skills profile based on survey responses"
                    className="h-full"
                  >
                    <ChartBars
                      data={[
                        { name: "Analytical", value: recommendationData.skillsAnalysis.analytical },
                        { name: "Creative", value: recommendationData.skillsAnalysis.creative },
                        { name: "Communication", value: recommendationData.skillsAnalysis.communication },
                        { name: "Teamwork", value: recommendationData.skillsAnalysis.teamwork },
                        { name: "Problem Solving", value: recommendationData.skillsAnalysis.problemSolving },
                      ]}
                      yAxisWidth={80}
                      showAnimation
                    >
                      {({ key, value, name, index, formattedValue, bar }) => (
                        <ChartBar
                          key={key}
                          x={bar.x}
                          y={bar.y}
                          width={bar.width}
                          height={bar.height}
                          rx={4}
                          className="fill-[#261FB3]"
                        >
                          <text
                            x={bar.x + bar.width / 2}
                            y={bar.y - 6}
                            textAnchor="middle"
                            className="fill-[#261FB3] font-medium text-xs"
                          >
                            {formattedValue}%
                          </text>
                        </ChartBar>
                      )}
                    </ChartBars>
                  </ChartContainer>
                </div>
                <div className="mt-6 text-sm text-muted-foreground">
                  <p>
                    Your skills profile shows strong analytical and problem-solving abilities, which align well with
                    technical and research-oriented fields. Your communication and teamwork skills are also
                    well-developed, making you suitable for collaborative environments.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      <div className="mt-8 flex justify-center gap-4">
        <Button variant="outline" className="gap-2">
          <Download className="h-4 w-4" />
          Download Results
        </Button>
        <Button variant="outline" className="gap-2">
          <Share2 className="h-4 w-4" />
          Share Results
        </Button>
      </div>
    </div>
  )
}
