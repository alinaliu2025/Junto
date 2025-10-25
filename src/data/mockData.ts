export interface Club {
  id: string;
  name: string;
  category: string;
  shortDescription: string;
  fullDescription: string;
  logo: string;
  coverImage: string;
  meetingTime: string;
  location: string;
  socials: {
    instagram?: string;
    facebook?: string;
    website?: string;
  };
  memberCount: number;
  events?: Event[];
  leaders?: Member[];
  eboard?: Member[];
  members?: Member[];
}

export interface Event {
  id: string;
  name: string;
  date: string;
  time: string;
  location: string;
  description: string;
}

export interface Member {
  id: string;
  name: string;
  role: string;
  avatar: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  avatar: string;
  joinedClubs: string[];
  year: string;
  major: string;
  hobbies: string[];
  extracurriculars: string[];
  classesTaken: string[];
  experiences: string[];
}

export const mockClubs: Club[] = [
  {
    id: "1",
    name: "OSU Photography Club",
    category: "Special Interest",
    shortDescription: "Capture moments and develop your photography skills with fellow enthusiasts.",
    fullDescription: "The OSU Photography Club is dedicated to students passionate about visual storytelling. Whether you're a beginner or experienced photographer, join us for workshops, photo walks, exhibitions, and critiques. We explore everything from landscape to portrait photography.",
    logo: "📷",
    coverImage: "https://images.unsplash.com/photo-1452587925148-ce544e77e70d?w=1200&h=400&fit=crop",
    meetingTime: "Thursdays at 6:00 PM",
    location: "Memorial Union, Room 204",
    socials: {
      instagram: "@osuphotoclub",
      website: "https://osuphotography.club"
    },
    memberCount: 87
  },
  {
    id: "2",
    name: "Women in Engineering",
    category: "Professional",
    shortDescription: "Empowering women in STEM through mentorship, networking, and professional development.",
    fullDescription: "Women in Engineering (WIE) at OSU creates a supportive community for women pursuing engineering degrees. We host industry speakers, organize career fairs, provide mentorship programs, and advocate for diversity in STEM fields. Open to all genders who support our mission.",
    logo: "⚙️",
    coverImage: "https://images.unsplash.com/photo-1573164713714-d95e436ab8d6?w=1200&h=400&fit=crop",
    meetingTime: "Tuesdays at 7:00 PM",
    location: "Kelley Engineering Center, Room 1003",
    socials: {
      instagram: "@osuwie",
      facebook: "OSU Women in Engineering"
    },
    memberCount: 124
  },
  {
    id: "3",
    name: "Alpha Kappa Lambda",
    category: "Greek Life",
    shortDescription: "Brotherhood, scholarship, and service since 1914.",
    fullDescription: "Alpha Kappa Lambda is a social fraternity focused on developing men of character through brotherhood, leadership, and community service. We maintain high academic standards, participate in philanthropic activities, and create lifelong bonds. Our chapter emphasizes personal growth and giving back to the Corvallis community.",
    logo: "ΑΚΛ",
    coverImage: "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=1200&h=400&fit=crop",
    meetingTime: "Mondays at 8:00 PM",
    location: "Chapter House, 234 NW Monroe Ave",
    socials: {
      instagram: "@akl_osu",
      website: "https://akl-osu.org"
    },
    memberCount: 65
  },
  {
    id: "4",
    name: "Salsa Club",
    category: "Cultural",
    shortDescription: "Learn salsa, bachata, and Latin dance in a fun, welcoming environment.",
    fullDescription: "The OSU Salsa Club brings Latin culture to campus through dance. No experience necessary! We teach salsa, bachata, and merengue to dancers of all levels. Join us for weekly lessons, social dancing, and special events. Make new friends while learning exciting new moves.",
    logo: "💃",
    coverImage: "https://images.unsplash.com/photo-1504609773096-104ff2c73ba4?w=1200&h=400&fit=crop",
    meetingTime: "Wednesdays at 7:30 PM",
    location: "Dixon Recreation Center, Studio B",
    socials: {
      instagram: "@osusalsa",
      facebook: "OSU Salsa Club"
    },
    memberCount: 93
  },
  {
    id: "5",
    name: "Club Soccer",
    category: "Sports",
    shortDescription: "Competitive soccer for students who love the beautiful game.",
    fullDescription: "OSU Club Soccer offers competitive and recreational soccer opportunities for students. We compete in regional tournaments, hold regular practices, and build team camaraderie. Whether you played in high school or just love the sport, there's a place for you on our roster.",
    logo: "⚽",
    coverImage: "https://images.unsplash.com/photo-1574629810360-7efbbe195018?w=1200&h=400&fit=crop",
    meetingTime: "Tuesdays & Thursdays at 5:00 PM",
    location: "OSU Soccer Fields",
    socials: {
      instagram: "@osuclubsoccer",
      website: "https://osusoccer.club"
    },
    memberCount: 112
  },
  {
    id: "6",
    name: "Investment Club",
    category: "Academic",
    shortDescription: "Learn about finance, investing, and market analysis through hands-on experience.",
    fullDescription: "The OSU Investment Club provides students with practical experience in financial markets and investment strategies. We manage a student portfolio, host industry professionals, analyze market trends, and prepare members for careers in finance. Great for business, economics, and finance majors.",
    logo: "📈",
    coverImage: "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=1200&h=400&fit=crop",
    meetingTime: "Wednesdays at 6:00 PM",
    location: "Austin Hall, Room 102",
    socials: {
      website: "https://osuinvestmentclub.org"
    },
    memberCount: 78
  },
  {
    id: "7",
    name: "Game Development Club",
    category: "Special Interest",
    shortDescription: "Create games, learn game design, and collaborate with fellow developers.",
    fullDescription: "The Game Development Club at OSU is for students interested in creating video games. We work on collaborative projects using Unity, Unreal Engine, and other tools. No coding experience required—we need artists, sound designers, writers, and more! Participate in game jams and build your portfolio.",
    logo: "🎮",
    coverImage: "https://images.unsplash.com/photo-1511512578047-dfb367046420?w=1200&h=400&fit=crop",
    meetingTime: "Fridays at 5:00 PM",
    location: "Kelley Engineering Center, Room 2005",
    socials: {
      instagram: "@osugamedev",
      website: "https://osugamedev.club"
    },
    memberCount: 145
  },
  {
    id: "8",
    name: "Kappa Alpha Theta",
    category: "Greek Life",
    shortDescription: "Leading women sorority focused on scholarship, leadership, and service.",
    fullDescription: "Kappa Alpha Theta is a women's fraternity that promotes intellectual excellence, leadership development, and community service. Our sisterhood values lifelong bonds, personal growth, and making a positive impact. We host philanthropy events, maintain high academic standards, and support each other's goals.",
    logo: "ΚΑΘ",
    coverImage: "https://images.unsplash.com/photo-1529333166437-7750a6dd5a70?w=1200&h=400&fit=crop",
    meetingTime: "Sundays at 7:00 PM",
    location: "Chapter House, 456 SW Harrison Blvd",
    socials: {
      instagram: "@theta_osu",
      website: "https://kappaalphathetaosu.org"
    },
    memberCount: 89
  }
];

export const mockUser: User = {
  id: "user1",
  name: "Alex Martinez",
  email: "alex.martinez@oregonstate.edu",
  avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Alex",
  joinedClubs: ["1", "4", "7"],
  year: "Junior",
  major: "Computer Science",
  hobbies: ["Photography", "Hiking", "Gaming", "Cooking"],
  extracurriculars: ["OSU Photography Club - Member", "Game Development Club - Project Lead", "Salsa Club - Active Participant"],
  classesTaken: ["CS 161 - Intro to Computer Science", "CS 271 - Computer Architecture", "CS 340 - Databases", "MTH 231 - Discrete Mathematics"],
  experiences: [
    "Software Engineering Intern at Intel (Summer 2024)",
    "Teaching Assistant for CS 161 (Fall 2023)",
    "Hackathon Winner - BeaverHacks 2023"
  ]
};

export const categories = [
  "All",
  "Greek Life",
  "Sports",
  "Cultural",
  "Academic",
  "Professional",
  "Special Interest"
];
