import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, Calendar, Users, Bell, Plus } from "lucide-react";
import { mockClubs } from "@/data/mockData";
import { CreateEventModal } from "@/components/CreateEventModal";
import { toast } from "sonner";

export default function ClubDashboardPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [isEventModalOpen, setIsEventModalOpen] = useState(false);
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

  const handleEventCreate = (event: any) => {
    toast.success(`Event "${event.name}" created successfully!`);
  };

  const stats = [
    { label: "Total Members", value: club.memberCount, icon: Users },
    { label: "Upcoming Events", value: "3", icon: Calendar },
    { label: "Announcements", value: "2", icon: Bell }
  ];

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="mb-6 flex items-center justify-between">
            <Button variant="ghost" onClick={() => navigate(`/club/${id}/member`)}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Club
            </Button>
            <Button variant="join" onClick={() => navigate(`/club/${id}/members`)}>
              <Users className="mr-2 h-4 w-4" />
              Manage Members
            </Button>
          </div>

          <div className="mb-8">
            <div className="flex items-center gap-4 mb-4">
              <span className="text-5xl">{club.logo}</span>
              <div>
                <h1 className="text-3xl font-bold">{club.name}</h1>
                <p className="text-muted-foreground">Club Dashboard</p>
              </div>
            </div>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {stats.map((stat) => (
              <Card key={stat.label}>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">{stat.label}</p>
                      <p className="text-3xl font-bold">{stat.value}</p>
                    </div>
                    <stat.icon className="h-8 w-8 text-primary" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Events Management */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="h-5 w-5 text-primary" />
                  Events
                </CardTitle>
                <Button size="sm" variant="join" onClick={() => setIsEventModalOpen(true)}>
                  <Plus className="h-4 w-4 mr-1" />
                  Create Event
                </Button>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="p-4 rounded-lg bg-secondary/50">
                  <h4 className="font-semibold mb-2">Weekly Meeting</h4>
                  <p className="text-sm text-muted-foreground mb-1">{club.meetingTime}</p>
                  <p className="text-sm text-muted-foreground">{club.location}</p>
                </div>
                <div className="p-4 rounded-lg bg-secondary/50">
                  <h4 className="font-semibold mb-2">Spring Showcase</h4>
                  <p className="text-sm text-muted-foreground mb-1">March 15, 2025 at 6:00 PM</p>
                  <p className="text-sm text-muted-foreground">Memorial Union Ballroom</p>
                </div>
                <div className="p-4 rounded-lg bg-secondary/50">
                  <h4 className="font-semibold mb-2">New Member Social</h4>
                  <p className="text-sm text-muted-foreground mb-1">March 22, 2025 at 7:00 PM</p>
                  <p className="text-sm text-muted-foreground">Chapter House</p>
                </div>
              </CardContent>
            </Card>

            {/* Announcements */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Bell className="h-5 w-5 text-primary" />
                  Recent Announcements
                </CardTitle>
                <Button size="sm" variant="outline">
                  <Plus className="h-4 w-4 mr-1" />
                  New
                </Button>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="p-4 rounded-lg bg-secondary/50">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold">Welcome New Members!</h4>
                    <span className="text-xs text-muted-foreground">2 days ago</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    We're excited to welcome 5 new members to the club this week.
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-secondary/50">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold">Meeting Location Change</h4>
                    <span className="text-xs text-muted-foreground">1 week ago</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    This week's meeting will be held in Room 305 instead of 204.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Quick Actions */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Button variant="outline" className="h-20 flex flex-col gap-2">
                  <Users className="h-6 w-6" />
                  <span className="text-sm">View Members</span>
                </Button>
                <Button variant="outline" className="h-20 flex flex-col gap-2">
                  <Bell className="h-6 w-6" />
                  <span className="text-sm">Send Alert</span>
                </Button>
                <Button variant="outline" className="h-20 flex flex-col gap-2">
                  <Calendar className="h-6 w-6" />
                  <span className="text-sm">Schedule</span>
                </Button>
                <Button variant="outline" className="h-20 flex flex-col gap-2">
                  <Plus className="h-6 w-6" />
                  <span className="text-sm">More</span>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>

      <CreateEventModal
        isOpen={isEventModalOpen}
        onClose={() => setIsEventModalOpen(false)}
        onSubmit={handleEventCreate}
      />
    </div>
  );
}
