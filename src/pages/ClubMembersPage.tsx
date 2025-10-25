import { useParams, useNavigate } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, User } from "lucide-react";
import { mockClubs } from "@/data/mockData";

export default function ClubMembersPage() {
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

  // Mock member data with roles
  const leaders = [
    { id: "1", name: "Sarah Johnson", role: "President", avatar: "SJ" },
    { id: "2", name: "Michael Chen", role: "Vice President", avatar: "MC" }
  ];

  const eboard = [
    { id: "3", name: "Emma Davis", role: "Treasurer", avatar: "ED" },
    { id: "4", name: "James Wilson", role: "Secretary", avatar: "JW" },
    { id: "5", name: "Olivia Brown", role: "Social Chair", avatar: "OB" },
    { id: "6", name: "Daniel Martinez", role: "Outreach Coordinator", avatar: "DM" }
  ];

  const members = [
    { id: "7", name: "Alex Martinez", role: "Member", avatar: "AM" },
    { id: "8", name: "Sophie Turner", role: "Member", avatar: "ST" },
    { id: "9", name: "Ryan Lee", role: "Member", avatar: "RL" },
    { id: "10", name: "Isabella Garcia", role: "Member", avatar: "IG" },
    { id: "11", name: "Ethan Park", role: "Member", avatar: "EP" },
    { id: "12", name: "Mia Anderson", role: "Member", avatar: "MA" },
    { id: "13", name: "Noah Thompson", role: "Member", avatar: "NT" },
    { id: "14", name: "Ava White", role: "Member", avatar: "AW" }
  ];

  const MemberCard = ({ member }: { member: typeof leaders[0] }) => (
    <Card className="hover:shadow-soft-lg transition-smooth cursor-pointer">
      <CardContent className="pt-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
            <span className="text-lg font-semibold text-primary">{member.avatar}</span>
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold truncate">{member.name}</h3>
            <p className="text-sm text-muted-foreground">{member.role}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1 p-6">
        <div className="max-w-7xl mx-auto">
          <Button variant="ghost" onClick={() => navigate(-1)} className="mb-6">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>

          <div className="mb-8">
            <div className="flex items-center gap-4 mb-2">
              <span className="text-4xl">{club.logo}</span>
              <h1 className="text-3xl font-bold">{club.name}</h1>
            </div>
            <p className="text-muted-foreground">
              {leaders.length + eboard.length + members.length} total members
            </p>
          </div>

          <div className="space-y-8">
            {/* Leaders Section */}
            <div>
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <User className="h-6 w-6 text-primary" />
                Leadership
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {leaders.map((leader) => (
                  <MemberCard key={leader.id} member={leader} />
                ))}
              </div>
            </div>

            {/* E-Board Section */}
            <div>
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <User className="h-6 w-6 text-primary" />
                Executive Board
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {eboard.map((member) => (
                  <MemberCard key={member.id} member={member} />
                ))}
              </div>
            </div>

            {/* General Members Section */}
            <div>
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <User className="h-6 w-6 text-primary" />
                General Members
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {members.map((member) => (
                  <MemberCard key={member.id} member={member} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
