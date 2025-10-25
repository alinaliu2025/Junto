import { useParams, useNavigate } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Bell, Calendar, Users } from "lucide-react";
import { mockClubs } from "@/data/mockData";

export default function MemberPrivatePage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const club = mockClubs.find((c) => c.id === id);

  if (!club) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Navbar />
        <div className="flex-1 flex items-center justify-center">
          <p className="text-muted-foreground">Club not found</p>
        </div>
      </div>
    );
  }

  // Mock member data
  const mockMembers = [
    "Sarah Johnson",
    "Michael Chen",
    "Emma Davis",
    "James Wilson",
    "Olivia Brown"
  ];

  const mockAnnouncements = [
    {
      id: 1,
      title: "Next Meeting Reminder",
      date: "2 days ago",
      content: "Don't forget our upcoming meeting this Thursday!"
    },
    {
      id: 2,
      title: "Welcome New Members",
      date: "5 days ago",
      content: "We're excited to welcome 5 new members to our club!"
    }
  ];

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1">
        {/* Hero Section */}
        <div
          className="h-48 bg-cover bg-center relative"
          style={{ backgroundImage: `url(${club.coverImage})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 to-black/60" />
          <Button
            variant="ghost"
            className="absolute top-4 left-4 text-white hover:bg-white/20"
            onClick={() => navigate("/dashboard")}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>
          <div className="absolute bottom-4 left-6 flex items-center gap-4">
            <span className="text-5xl">{club.logo}</span>
            <h1 className="text-3xl font-bold text-white">{club.name}</h1>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Announcements */}
              <Card className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Bell className="h-5 w-5 text-primary" />
                  <h2 className="text-xl font-semibold">Announcements</h2>
                </div>
                <div className="space-y-4">
                  {mockAnnouncements.map((announcement) => (
                    <div
                      key={announcement.id}
                      className="p-4 rounded-lg bg-secondary/50"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="font-semibold">{announcement.title}</h3>
                        <span className="text-xs text-muted-foreground">
                          {announcement.date}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {announcement.content}
                      </p>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Upcoming Events */}
              <Card className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Calendar className="h-5 w-5 text-primary" />
                  <h2 className="text-xl font-semibold">Upcoming Events</h2>
                </div>
                <div className="p-4 rounded-lg bg-secondary/50">
                  <h3 className="font-semibold mb-2">Regular Meeting</h3>
                  <p className="text-sm text-muted-foreground mb-1">
                    {club.meetingTime}
                  </p>
                  <p className="text-sm text-muted-foreground">{club.location}</p>
                </div>
              </Card>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Quick Actions (for club leaders) */}
              <Card className="p-6">
                <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
                <div className="space-y-2">
                  <Button 
                    variant="outline" 
                    className="w-full justify-start"
                    onClick={() => navigate(`/club/${id}/dashboard`)}
                  >
                    Club Dashboard
                  </Button>
                  <Button 
                    variant="outline" 
                    className="w-full justify-start"
                    onClick={() => navigate(`/club/${id}/members`)}
                  >
                    <Users className="h-4 w-4 mr-2" />
                    View All Members
                  </Button>
                </div>
              </Card>

              {/* Members List */}
              <Card className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Users className="h-5 w-5 text-primary" />
                    <h2 className="text-lg font-semibold">Members</h2>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => navigate(`/club/${id}/members`)}
                    className="text-primary hover:text-primary"
                  >
                    View All
                  </Button>
                </div>
                <div className="space-y-2">
                  {mockMembers.map((member, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-3 p-2 rounded-lg hover:bg-secondary/50 transition-base"
                    >
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-sm font-medium">
                        {member.split(" ").map(n => n[0]).join("")}
                      </div>
                      <span className="text-sm">{member}</span>
                    </div>
                  ))}
                  <p className="text-xs text-muted-foreground text-center pt-2">
                    +{club.memberCount - mockMembers.length} more members
                  </p>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
