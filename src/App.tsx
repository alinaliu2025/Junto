import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import SignInPage from "./pages/SignInPage";
import DiscoveryPage from "./pages/DiscoveryPage";
import ClubProfilePage from "./pages/ClubProfilePage";
import MemberDashboardPage from "./pages/MemberDashboardPage";
import MemberPrivatePage from "./pages/MemberPrivatePage";
import CreateClubPage from "./pages/CreateClubPage";
import ClubDashboardPage from "./pages/ClubDashboardPage";
import UserProfilePage from "./pages/UserProfilePage";
import ClubMembersPage from "./pages/ClubMembersPage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<SignInPage />} />
          <Route path="/discovery" element={<DiscoveryPage />} />
          <Route path="/club/:id" element={<ClubProfilePage />} />
          <Route path="/club/:id/member" element={<MemberPrivatePage />} />
          <Route path="/club/:id/dashboard" element={<ClubDashboardPage />} />
          <Route path="/club/:id/members" element={<ClubMembersPage />} />
          <Route path="/dashboard" element={<MemberDashboardPage />} />
          <Route path="/create-club" element={<CreateClubPage />} />
          <Route path="/profile" element={<UserProfilePage />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
