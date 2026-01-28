import type { User, AdminStats, UserActivity } from "@/types";

export const mockUsers: User[] = [
  {
    id: "user-001",
    email: "dr.rahman@dhaka-hospital.bd",
    name: "Dr. Rahim Rahman",
    role: "admin",
    organization: "Dhaka Medical College Hospital",
    createdAt: "2023-06-15T00:00:00Z",
    lastLogin: "2024-01-25T09:30:00Z",
  },
  {
    id: "user-002",
    email: "dr.chowdhury@chittagong-lab.bd",
    name: "Dr. Fatima Chowdhury",
    role: "user",
    organization: "Chittagong Diagnostic Lab",
    createdAt: "2023-08-20T00:00:00Z",
    lastLogin: "2024-01-24T14:15:00Z",
  },
  {
    id: "user-003",
    email: "dr.islam@icddrb.org",
    name: "Dr. Kamal Islam",
    role: "user",
    organization: "icddr,b",
    createdAt: "2023-09-10T00:00:00Z",
    lastLogin: "2024-01-25T08:45:00Z",
  },
  {
    id: "user-004",
    email: "lab.tech@bsmmu.edu.bd",
    name: "Nasreen Begum",
    role: "lab_technician",
    organization: "BSMMU",
    createdAt: "2023-10-05T00:00:00Z",
    lastLogin: "2024-01-23T16:30:00Z",
  },
  {
    id: "user-005",
    email: "dr.khan@sylhet-hospital.bd",
    name: "Dr. Amir Khan",
    role: "user",
    organization: "Sylhet MAG Osmani Medical College",
    createdAt: "2023-11-12T00:00:00Z",
    lastLogin: "2024-01-24T11:00:00Z",
  },
  {
    id: "user-006",
    email: "dr.sultana@rajshahi-lab.bd",
    name: "Dr. Ayesha Sultana",
    role: "user",
    organization: "Rajshahi Medical College Hospital",
    createdAt: "2023-12-01T00:00:00Z",
    lastLogin: "2024-01-25T10:15:00Z",
  },
  {
    id: "user-007",
    email: "dr.hossain@khulna-clinic.bd",
    name: "Dr. Mizanur Hossain",
    role: "user",
    organization: "Khulna City Clinic",
    createdAt: "2024-01-05T00:00:00Z",
    lastLogin: "2024-01-22T09:00:00Z",
  },
  {
    id: "user-008",
    email: "research.ahmed@nih.bd",
    name: "Dr. Shahadat Ahmed",
    role: "admin",
    organization: "National Institute of Health",
    createdAt: "2023-07-20T00:00:00Z",
    lastLogin: "2024-01-25T07:45:00Z",
  },
];

export const currentUser: User = mockUsers[0];

export const adminStats: AdminStats = {
  totalUsers: mockUsers.length,
  activeUsers: mockUsers.filter(u => {
    const lastLogin = new Date(u.lastLogin || "");
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    return lastLogin > sevenDaysAgo;
  }).length,
  totalPredictions: 156,
  predictionsToday: 12,
  storageUsed: 2.4, // GB
  storageLimit: 10, // GB
};

export const recentActivity: UserActivity[] = [
  {
    userId: "user-001",
    userName: "Dr. Rahim Rahman",
    action: "Uploaded sample",
    timestamp: "2024-01-25T10:30:00Z",
    details: "TB-2024-0128-A.fasta",
  },
  {
    userId: "user-002",
    userName: "Dr. Fatima Chowdhury",
    action: "Viewed results",
    timestamp: "2024-01-25T10:15:00Z",
    details: "EC-2024-0127-B",
  },
  {
    userId: "user-003",
    userName: "Dr. Kamal Islam",
    action: "Downloaded report",
    timestamp: "2024-01-25T09:45:00Z",
    details: "SA-2024-0126-C.pdf",
  },
  {
    userId: "user-006",
    userName: "Dr. Ayesha Sultana",
    action: "Uploaded sample",
    timestamp: "2024-01-25T09:30:00Z",
    details: "VC-2024-0121-H.fastq",
  },
  {
    userId: "user-008",
    userName: "Dr. Shahadat Ahmed",
    action: "Added new user",
    timestamp: "2024-01-25T08:00:00Z",
    details: "dr.new@hospital.bd",
  },
];
