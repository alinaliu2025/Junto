import { useNavigate } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, User, Briefcase, Book, Star, Award } from "lucide-react";
import { mockUser } from "@/data/mockData";

export default function UserProfilePage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1 p-6">
        <div className="max-w-5xl mx-auto">
          <Button variant="ghost" onClick={() => navigate(-1)} className="mb-6">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>

          {/* Profile Header */}
          <Card className="mb-6">
            <CardContent className="pt-6">
              <div className="flex items-start gap-6">
                <div className="w-24 h-24 rounded-full bg-primary/10 flex items-center justify-center">
                  <User className="h-12 w-12 text-primary" />
                </div>
                <div className="flex-1">
                  <h1 className="text-3xl font-bold mb-2">{mockUser.name}</h1>
                  <p className="text-muted-foreground mb-1">{mockUser.email}</p>
                  <div className="flex gap-4 text-sm mt-3">
                    <span className="text-foreground">
                      <span className="font-semibold">Year:</span> {mockUser.year}
                    </span>
                    <span className="text-foreground">
                      <span className="font-semibold">Major:</span> {mockUser.major}
                    </span>
                  </div>
                </div>
                <Button variant="outline">Edit Profile</Button>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Hobbies & Interests */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Star className="h-5 w-5 text-primary" />
                  Hobbies & Interests
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {mockUser.hobbies.map((hobby, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-secondary rounded-full text-sm"
                    >
                      {hobby}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Extracurriculars */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="h-5 w-5 text-primary" />
                  Extracurriculars
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {mockUser.extracurriculars.map((activity, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-primary mt-1">•</span>
                      <span className="text-sm">{activity}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Classes Taken */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Book className="h-5 w-5 text-primary" />
                  Notable Classes
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {mockUser.classesTaken.map((course, index) => (
                    <div
                      key={index}
                      className="p-3 bg-secondary/50 rounded-lg text-sm"
                    >
                      {course}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Experiences */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Briefcase className="h-5 w-5 text-primary" />
                  Experience & Accomplishments
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {mockUser.experiences.map((experience, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-primary mt-1">•</span>
                      <span className="text-sm">{experience}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
