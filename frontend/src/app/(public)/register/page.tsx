import { RegisterForm } from "@/components/features/auth/register-form";

export const metadata = {
  title: "Create Account - DeepAMR",
  description: "Create a new DeepAMR account to start predicting antimicrobial resistance",
};

export default function RegisterPage() {
  return <RegisterForm />;
}
